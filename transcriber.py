
import logging
import os
import json
import platform
import re
import tempfile
import time
from difflib import SequenceMatcher

from dotenv import load_dotenv
from faster_whisper import BatchedInferencePipeline, WhisperModel
from moviepy.editor import VideoFileClip
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

load_dotenv()

logging.basicConfig(level=logging.INFO)

SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
SPEECHMATICS_API_BASE = os.getenv("SPEECHMATICS_API_BASE", "https://eu1.asr.api.speechmatics.com/v2").rstrip("/")
SPEECHMATICS_LANGUAGE = os.getenv("SPEECHMATICS_LANGUAGE", "auto")
SPEECHMATICS_MODEL = os.getenv("SPEECHMATICS_MODEL", "enhanced")
SPEECHMATICS_EXPECTED_LANGUAGES = [
    language.strip()
    for language in os.getenv("SPEECHMATICS_EXPECTED_LANGUAGES", "de,en").split(",")
    if language.strip()
]
SPEECHMATICS_POLL_SECONDS = float(os.getenv("SPEECHMATICS_POLL_SECONDS", "5"))
SPEECHMATICS_TIMEOUT_SECONDS = int(os.getenv("SPEECHMATICS_TIMEOUT_SECONDS", "900"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "24"))
WHISPER_COMPUTE_TYPES = ("int8_float16", "float16", "int8")
MAX_WORDS_PER_LOCAL_CAPTION = 3
MAX_LOCAL_CAPTION_CHARS = 24
MAX_LOCAL_LINES_PER_CAPTION = 2
MAX_LOCAL_SHORT_WORDS_PER_LINE = 2
LONG_LOCAL_WORD_MIN_CHARS = 8
LOCAL_TRANSCRIPT_DELAY_START_SECONDS = 17.0
LOCAL_TRANSCRIPT_DELAY_SECONDS = 0.0
LOCAL_TRANSCRIPT_DELAY_VIDEO_NAMES = set()
TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?)\s*(?:-->|-|–|—)\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?)\s*(?P<text>.*)"
)
INLINE_TIMESTAMP_RE = re.compile(
    r"\(?(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?)\)?"
)


def _detect_cpu_threads():
    configured_threads = os.getenv("WHISPER_CPU_THREADS")
    if configured_threads:
        return max(1, int(configured_threads))

    logical_threads = os.cpu_count() or 8
    if platform.system() == "Windows" and logical_threads >= 16:
        return max(8, logical_threads // 2)
    return logical_threads


def _directml_available():
    try:
        import onnxruntime as ort
    except ImportError:
        return False

    providers = ort.get_available_providers()
    if "DmlExecutionProvider" in providers:
        logging.info("ONNX Runtime DirectML provider is available: %s", providers)
        return True

    logging.info("ONNX Runtime DirectML provider is not available. Providers: %s", providers)
    return False


def _load_whisper_model(compute_type, cpu_threads):
    logging.info(
        "Loading faster-whisper model '%s' on CPU with compute_type=%s, cpu_threads=%s...",
        WHISPER_MODEL_SIZE,
        compute_type,
        cpu_threads,
    )
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
    )


def _transcribe_with_fastest_backend(audio_path):
    cpu_threads = _detect_cpu_threads()
    directml_ready = _directml_available()
    if directml_ready:
        logging.info("DirectML is available for ONNX Runtime; faster-whisper will still use CTranslate2 CPU unless an ONNX Whisper backend is added.")

    last_error = None
    for compute_type in WHISPER_COMPUTE_TYPES:
        try:
            model = _load_whisper_model(compute_type, cpu_threads)
            batched_model = BatchedInferencePipeline(model=model)

            logging.info(
                "Starting batched transcription with %s CPU threads, compute_type=%s, batch_size=%s...",
                cpu_threads,
                compute_type,
                WHISPER_BATCH_SIZE,
            )
            segments, _ = batched_model.transcribe(
                audio_path,
                word_timestamps=True,
                vad_filter=True,
                batch_size=WHISPER_BATCH_SIZE,
                beam_size=1,
            )

            word_level_transcript = []
            for segment in segments:
                for word in segment.words:
                    word_level_transcript.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    })
            return word_level_transcript
        except Exception as exc:
            last_error = exc
            logging.warning("Transcription failed with compute_type=%s; trying fallback. Error: %s", compute_type, exc)

    raise RuntimeError(f"Could not transcribe with any faster-whisper compute type: {last_error}")


def _speechmatics_headers():
    return {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}


def _speechmatics_job_config():
    transcription_config = {
        "language": SPEECHMATICS_LANGUAGE,
        "model": SPEECHMATICS_MODEL,
        "diarization": "speaker",
        "punctuation_overrides": {"permitted_marks": ["?", ".", "!", ","]},
    }
    config = {
        "type": "transcription",
        "transcription_config": transcription_config,
    }

    if SPEECHMATICS_LANGUAGE == "auto":
        config["language_identification_config"] = {"low_confidence_action": "allow"}
        if SPEECHMATICS_EXPECTED_LANGUAGES:
            config["language_identification_config"]["expected_languages"] = SPEECHMATICS_EXPECTED_LANGUAGES

    return config


def _speechmatics_submit_job(audio_path):
    logging.info("Submitting audio to Speechmatics Batch API...")
    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            f"{SPEECHMATICS_API_BASE}/jobs",
            headers=_speechmatics_headers(),
            files={"data_file": (os.path.basename(audio_path), audio_file)},
            data={"config": json.dumps(_speechmatics_job_config())},
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("id") or payload.get("job", {}).get("id")
    if not job_id:
        raise RuntimeError(f"Speechmatics did not return a job id: {payload}")
    logging.info("Speechmatics job submitted: %s", job_id)
    return job_id


def _speechmatics_wait_for_job(job_id):
    started_at = time.time()
    while True:
        response = requests.get(
            f"{SPEECHMATICS_API_BASE}/jobs/{job_id}",
            headers=_speechmatics_headers(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        job = payload.get("job", payload)
        status = job.get("status")
        logging.info("Speechmatics job %s status: %s", job_id, status)

        if status == "done":
            return
        if status == "rejected":
            errors = job.get("errors") or payload.get("errors") or []
            raise RuntimeError(f"Speechmatics rejected job {job_id}: {errors or payload}")
        if time.time() - started_at > SPEECHMATICS_TIMEOUT_SECONDS:
            raise TimeoutError(f"Speechmatics job {job_id} did not finish within {SPEECHMATICS_TIMEOUT_SECONDS}s.")

        time.sleep(SPEECHMATICS_POLL_SECONDS)


def _parse_speechmatics_transcript(payload):
    word_level_transcript = []
    block_index = -1
    previous_end = None

    for result in payload.get("results", []):
        alternatives = result.get("alternatives") or []
        if not alternatives:
            continue

        content = alternatives[0].get("content", "").strip()
        if not content:
            continue

        if result.get("type") == "punctuation":
            if word_level_transcript and content in {".", "!", "?", ",", ";", ":"}:
                word_level_transcript[-1]["word"] = f'{word_level_transcript[-1]["word"]}{content}'
            continue

        if result.get("type") != "word":
            continue

        start = result.get("start_time")
        end = result.get("end_time")
        if start is None or end is None:
            continue

        start = float(start)
        end = float(end)
        if previous_end is None or start - previous_end > 0.8:
            block_index += 1
        previous_end = end

        word_level_transcript.append({
            "word": content,
            "start": start,
            "end": end,
            "block_index": block_index,
            "speaker": result.get("speaker"),
        })

    return word_level_transcript


def _speechmatics_fetch_transcript(job_id):
    response = requests.get(
        f"{SPEECHMATICS_API_BASE}/jobs/{job_id}/transcript",
        headers=_speechmatics_headers(),
        params={"format": "json-v2"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    word_level_transcript = _parse_speechmatics_transcript(payload)
    if not word_level_transcript:
        raise RuntimeError("Speechmatics returned no word-level timestamps.")
    return word_level_transcript


def _transcribe_with_speechmatics(audio_path):
    if not SPEECHMATICS_API_KEY:
        return None

    job_id = _speechmatics_submit_job(audio_path)
    _speechmatics_wait_for_job(job_id)
    return _speechmatics_fetch_transcript(job_id)


def extract_audio(video_path, audio_path="temp_audio.mp3"):
    """Extracts audio from a video file and saves it."""
    if not os.path.exists(video_path):
        logging.error(f"Video file not found at: {video_path}")
        return None
    try:
        logging.info(f"Extracting audio from {video_path}...")
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, codec='mp3')
        audio_clip.close()
        video_clip.close()
        logging.info(f"Audio extracted and saved to {audio_path}")
        return audio_path
    except Exception as e:
        logging.error(f"Failed to extract audio: {e}")
        return None


def find_sidecar_transcript(video_path):
    raw_video_dir = os.path.dirname(video_path)
    fixed_transcript_path = os.path.join(raw_video_dir, "untertitel.txt")
    if os.path.exists(fixed_transcript_path) and os.path.getsize(fixed_transcript_path) > 0:
        return fixed_transcript_path

    base_path, _ = os.path.splitext(video_path)
    transcript_path = f"{base_path}.txt"
    if os.path.exists(transcript_path) and os.path.getsize(transcript_path) > 0:
        return transcript_path
    return None


def _parse_timestamp(timestamp):
    timestamp = timestamp.strip().replace(",", ".")
    parts = timestamp.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp format: {timestamp}")


def _parse_timestamped_blocks(text):
    blocks = []
    pending = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.isdigit():
            continue

        line = line.strip("[]")
        match = TIMESTAMP_RE.match(line)
        if match:
            if pending and pending["text"]:
                blocks.append(pending)
            pending = {
                "start": _parse_timestamp(match.group("start")),
                "end": _parse_timestamp(match.group("end")),
                "text": match.group("text").strip(),
            }
            continue

        if pending:
            pending["text"] = f'{pending["text"]} {line}'.strip()

    if pending and pending["text"]:
        blocks.append(pending)

    return blocks


def _parse_inline_timestamp_blocks(text, video_duration=None):
    matches = list(INLINE_TIMESTAMP_RE.finditer(text))
    if not matches:
        return []

    blocks = []
    for index, match in enumerate(matches):
        start = _parse_timestamp(match.group("time"))
        text_start = match.end()
        text_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block_text = text[text_start:text_end].strip()
        if not block_text:
            continue

        if index + 1 < len(matches):
            end = _parse_timestamp(matches[index + 1].group("time"))
        elif video_duration and video_duration > start:
            end = float(video_duration)
        else:
            end = start + 5.0

        blocks.append({
            "start": start,
            "end": end,
            "text": block_text,
        })

    return blocks


def _split_words_into_chunks(words):
    chunks = []
    current = []
    for word in words:
        candidate = [*current, word]
        if current and not _fits_local_caption_constraints(candidate):
            chunks.append(current)
            current = [word]
        else:
            current = candidate

        if len(current) >= 2 and str(word).endswith((".", "!", "?", ",", ";", ":")):
            chunks.append(current)
            current = []

    if current:
        if chunks and len(current) == 1 and _fits_local_caption_constraints([*chunks[-1], *current]):
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    return chunks


def _normalize_local_word(word):
    normalized = str(word).lower()
    normalized = (
        normalized
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _is_local_long_word(word):
    return len(_normalize_local_word(word)) >= LONG_LOCAL_WORD_MIN_CHARS


def _local_caption_lines(words):
    lines = []
    current_line = []
    for word in words:
        clean_word = str(word).strip()[:MAX_LOCAL_CAPTION_CHARS]
        if not clean_word:
            continue

        if _is_local_long_word(clean_word):
            if current_line:
                lines.append(current_line)
                current_line = []
            lines.append([clean_word])
            continue

        if len(current_line) >= MAX_LOCAL_SHORT_WORDS_PER_LINE or any(_is_local_long_word(existing) for existing in current_line):
            lines.append(current_line)
            current_line = [clean_word]
        else:
            current_line.append(clean_word)

    if current_line:
        lines.append(current_line)

    return lines


def _fits_local_caption_constraints(words):
    text = " ".join(str(word).strip()[:MAX_LOCAL_CAPTION_CHARS] for word in words if str(word).strip())
    if len(text) > MAX_LOCAL_CAPTION_CHARS:
        return False

    lines = _local_caption_lines(words)
    if len(lines) > MAX_LOCAL_LINES_PER_CAPTION:
        return False

    for line in lines:
        if any(_is_local_long_word(word) for word in line):
            if len(line) != 1:
                return False
        elif len(line) > MAX_LOCAL_SHORT_WORDS_PER_LINE:
            return False

    return True


def _local_transcript_delay_for_video(video_path):
    if not video_path:
        return 0.0
    video_name = os.path.basename(video_path).lower()
    if video_name in LOCAL_TRANSCRIPT_DELAY_VIDEO_NAMES:
        return LOCAL_TRANSCRIPT_DELAY_SECONDS
    return 0.0


def _blocks_to_word_transcript(blocks, video_duration=None, video_path=None):
    transcript_delay = _local_transcript_delay_for_video(video_path)
    word_transcript = []
    for block_index, block in enumerate(blocks):
        words = [word for word in block["text"].split() if word.strip()]
        if not words:
            continue

        start = float(block["start"])
        end = float(block["end"])
        if transcript_delay and start >= LOCAL_TRANSCRIPT_DELAY_START_SECONDS:
            start += transcript_delay
            end += transcript_delay
            if video_duration:
                end = min(end, float(video_duration))
            if start >= end:
                continue
        if end <= start:
            end = start + max(0.25, len(words) * 0.25)

        chunks = _split_words_into_chunks(words)
        total_chunks = len(chunks)
        duration = end - start
        cursor = start

        for index, chunk in enumerate(chunks):
            # Keep the user's timestamp block as the hard timing envelope.
            # If the block must be split by visual constraints, divide that envelope evenly.
            chunk_start = start + duration * (index / total_chunks)
            chunk_end = end if index == total_chunks - 1 else start + duration * ((index + 1) / total_chunks)
            cursor = chunk_start
            chunk_duration = chunk_end - chunk_start
            word_duration = chunk_duration / len(chunk)
            for word in chunk:
                word_start = cursor
                word_end = min(chunk_end, cursor + word_duration)
                word_transcript.append({
                    "word": word,
                    "start": word_start,
                    "end": word_end,
                    "block_index": block_index,
                })
                cursor = word_end

    return word_transcript


def transcribe_from_text_file(transcript_path, video_path=None):
    logging.info("Using local transcript file instead of Whisper: %s", transcript_path)
    with open(transcript_path, "r", encoding="utf-8-sig") as transcript_file:
        text = transcript_file.read()

    video_duration = None
    if video_path and os.path.exists(video_path):
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration
        video_clip.close()

    blocks = _parse_timestamped_blocks(text)
    if not blocks:
        blocks = _parse_inline_timestamp_blocks(text, video_duration)
    if not blocks:
        logging.error("No timestamped subtitle blocks found in %s.", transcript_path)
        return None

    word_transcript = _blocks_to_word_transcript(blocks, video_duration, video_path)
    logging.info("Parsed %s subtitle blocks into %s word timestamps.", len(blocks), len(word_transcript))
    return word_transcript


def transcribe_from_reference_file(transcript_path, video_path):
    """Uses an SRT/VTT as text reference, then aligns it to local audio word timings."""
    logging.info("Using subtitle file as reference text with internal audio synchronization: %s", transcript_path)
    blocks = _reference_blocks_from_file(transcript_path, video_path)
    if not blocks:
        logging.warning("Reference subtitle parsing failed; falling back to direct subtitle timings.")
        return transcribe_from_text_file(transcript_path, video_path)

    reference_words = _reference_words_from_blocks(blocks)
    if not reference_words:
        logging.warning("Reference subtitle contains no words; falling back to direct subtitle timings.")
        return transcribe_from_text_file(transcript_path, video_path)

    temp_audio_path = os.path.join(tempfile.gettempdir(), f"legmon_reference_sync_{int(time.time() * 1000)}.mp3")
    audio_path = extract_audio(video_path, temp_audio_path)
    if not audio_path:
        logging.warning("Could not extract audio for reference sync; falling back to direct subtitle timings.")
        return transcribe_from_text_file(transcript_path, video_path)

    audio_transcript = transcribe_audio(audio_path)
    if not audio_transcript:
        logging.warning("Internal audio transcription failed; falling back to direct subtitle timings.")
        return transcribe_from_text_file(transcript_path, video_path)

    aligned = _align_reference_words_to_audio(reference_words, audio_transcript)
    logging.info(
        "Aligned %s reference words to %s internally transcribed words.",
        len(aligned),
        len(audio_transcript),
    )
    return aligned


def _reference_blocks_from_file(transcript_path, video_path=None):
    with open(transcript_path, "r", encoding="utf-8-sig") as transcript_file:
        text = transcript_file.read()

    video_duration = None
    if video_path and os.path.exists(video_path):
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration
        video_clip.close()

    blocks = _parse_timestamped_blocks(text)
    if not blocks:
        blocks = _parse_inline_timestamp_blocks(text, video_duration)
    return blocks


def _reference_words_from_blocks(blocks):
    reference_words = []
    for block_index, block in enumerate(blocks):
        for word in str(block.get("text", "")).split():
            clean_word = word.strip()
            if not clean_word:
                continue
            reference_words.append({
                "word": clean_word,
                "block_index": block_index,
                "reference_start": float(block.get("start", 0.0)),
                "reference_end": float(block.get("end", block.get("start", 0.0))),
            })
    return reference_words


def _align_reference_words_to_audio(reference_words, audio_transcript):
    reference_tokens = [_normalize_local_word(item["word"]) for item in reference_words]
    audio_tokens = [_normalize_local_word(item.get("word", "")) for item in audio_transcript]
    matcher = SequenceMatcher(None, reference_tokens, audio_tokens, autojunk=False)
    matched_audio_indices = {}

    for tag, ref_start, ref_end, audio_start, audio_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(ref_end - ref_start):
            matched_audio_indices[ref_start + offset] = audio_start + offset

    aligned = []
    for index, reference_word in enumerate(reference_words):
        audio_index = matched_audio_indices.get(index)
        if audio_index is not None and audio_index < len(audio_transcript):
            audio_word = audio_transcript[audio_index]
            start = float(audio_word.get("start", reference_word["reference_start"]))
            end = float(audio_word.get("end", start + 0.25))
            speaker = audio_word.get("speaker")
        else:
            start, end = _interpolated_reference_word_time(index, reference_words, audio_transcript, matched_audio_indices)
            speaker = _interpolated_reference_speaker(index, audio_transcript, matched_audio_indices)

        if aligned:
            start = max(start, aligned[-1]["end"])
        if end <= start:
            end = start + 0.18

        aligned_word = {
            "word": reference_word["word"],
            "start": start,
            "end": end,
            "block_index": reference_word["block_index"],
        }
        if speaker:
            aligned_word["speaker"] = speaker
        aligned.append(aligned_word)

    return aligned


def _interpolated_reference_speaker(index, audio_transcript, matched_audio_indices):
    previous_ref = next((candidate for candidate in range(index - 1, -1, -1) if candidate in matched_audio_indices), None)
    next_ref = next((candidate for candidate in range(index + 1, index + 8) if candidate in matched_audio_indices), None)

    if previous_ref is not None:
        speaker = audio_transcript[matched_audio_indices[previous_ref]].get("speaker")
        if speaker:
            return speaker
    if next_ref is not None:
        speaker = audio_transcript[matched_audio_indices[next_ref]].get("speaker")
        if speaker:
            return speaker
    return None


def _interpolated_reference_word_time(index, reference_words, audio_transcript, matched_audio_indices):
    previous_ref = next((candidate for candidate in range(index - 1, -1, -1) if candidate in matched_audio_indices), None)
    next_ref = next((candidate for candidate in range(index + 1, len(reference_words)) if candidate in matched_audio_indices), None)

    if previous_ref is not None and next_ref is not None:
        previous_audio = audio_transcript[matched_audio_indices[previous_ref]]
        next_audio = audio_transcript[matched_audio_indices[next_ref]]
        start_bound = float(previous_audio.get("end", previous_audio.get("start", 0.0)))
        end_bound = float(next_audio.get("start", start_bound + 0.25))
        slots = max(1, next_ref - previous_ref)
        step = max(0.18, (end_bound - start_bound) / slots)
        start = start_bound + step * (index - previous_ref - 1)
        return start, min(end_bound, start + step * 0.9)

    if previous_ref is not None:
        previous_audio = audio_transcript[matched_audio_indices[previous_ref]]
        start = float(previous_audio.get("end", previous_audio.get("start", 0.0))) + 0.24 * (index - previous_ref - 1)
        return start, start + 0.22

    if next_ref is not None:
        next_audio = audio_transcript[matched_audio_indices[next_ref]]
        end = max(0.18, float(next_audio.get("start", 0.25)) - 0.24 * (next_ref - index - 1))
        return max(0.0, end - 0.22), end

    reference_word = reference_words[index]
    block_words = [
        word for word in reference_words
        if word["block_index"] == reference_word["block_index"]
    ]
    position = block_words.index(reference_word)
    duration = max(0.25, reference_word["reference_end"] - reference_word["reference_start"])
    step = duration / max(1, len(block_words))
    start = reference_word["reference_start"] + position * step
    return start, start + step * 0.9


def transcribe_audio(audio_path):
    """Transcribes audio and returns word-level timestamps."""
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found at: {audio_path}")
        return None
    try:
        try:
            word_level_transcript = _transcribe_with_speechmatics(audio_path)
        except Exception as speechmatics_error:
            logging.warning("Speechmatics transcription failed; falling back to faster-whisper. Error: %s", speechmatics_error)
            word_level_transcript = None

        if not word_level_transcript:
            word_level_transcript = _transcribe_with_fastest_backend(audio_path)
        
        logging.info("Transcription complete.")
        # Clean up the temporary audio file
        os.remove(audio_path)
        logging.info(f"Removed temporary audio file: {audio_path}")
        
        return word_level_transcript
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        return None
