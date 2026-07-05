import logging
import os
import re
import json
import bisect

import PIL.Image

if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = 1

from moviepy.audio.fx.all import audio_normalize
from moviepy.editor import AudioFileClip, CompositeAudioClip, CompositeVideoClip, ImageClip, VideoFileClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_BLACK_PATH = r"C:\Users\alexa\AppData\Local\Microsoft\Windows\Fonts\Montserrat-Black.ttf"
FONT_BOLD_PATH = os.path.join(BASE_DIR, "assets", "fonts", "Montserrat-Bold.ttf")
FONT_REGULAR_PATH = os.path.join(BASE_DIR, "assets", "fonts", "Montserrat-Regular.ttf")
EMOJI_MAPPING_PATH = os.path.join(BASE_DIR, "emoji_mapping.json")
EMOJI_ASSET_DIR = r"C:\Users\alexa\OneDrive\Desktop\AI-Video-Editor\assets"

CAPTION_FONT_SIZE = 75
CAPTION_COLOR = (255, 255, 255, 255)
CAPTION_STROKE_COLOR = (0, 0, 0, 255)
CAPTION_STROKE_WIDTH = 5
CAPTION_Y_RATIO = 0.49
CAPTION_SAFE_WIDTH_RATIO = 0.82
EMOJI_Y_RATIO = 0.60
EMOJI_HEIGHT = 140
EMOJI_SIZE = 240
HANDSHAKE_EMOJI_NAME = "handschlag-removebg-preview.png"
HANDSHAKE_EMOJI_SIZE = (180, 120)
THINKING_EMOJI_NAME = "thinking-removebg-preview.png"
THINKING_EMOJI_TOKENS = {"denken", "nachdenken", "ueberlegen", "gruebeln"}
EMOJI_GAP_PX = 20
NO_EMOJI_START_SECONDS_BY_VIDEO = {
    "roh ohne untertigtel.mp4": 4.0,
}
MIN_WORDS_PER_CAPTION = 2
MAX_LINES_PER_CAPTION = 2
MAX_SHORT_WORDS_PER_LINE = 2
MAX_CAPTION_CHARS = 24
LONG_WORD_MIN_CHARS = 8
MAX_CAPTION_DURATION = 1.15
MIN_SECONDS_BETWEEN_EMOJIS = 4.0
MAX_SAME_EMOJI_PER_VIDEO = 2
MIN_SECONDS_BETWEEN_HIGHLIGHTS = 2.0
HIGHLIGHT_EVERY_NTH_CAPTION = 2
IMPORTANCE_ZOOM_SCALE = 1.55
ANSWER_ZOOM_Y_CENTER_RATIO = 0.52
ANSWER_ZOOM_LEFT_CENTER_RATIO = 0.34
ANSWER_ZOOM_RIGHT_CENTER_RATIO = 0.66
IMPORTANCE_ZOOM_MIN_DURATION = 0.85
IMPORTANCE_ZOOM_MAX_DURATION = 1.20
MIN_SECONDS_BETWEEN_IMPORTANCE_ZOOMS = 3.0
IMPORTANCE_ZOOM_SFX_VOLUME = 0.55
OPENING_HIT_SFX_VOLUME = 0.45
KEYWORD_PING_SFX_VOLUME = 0.72
TOPIC_WOSH_SFX_VOLUME = 1.35
TOPIC_RISER_SFX_VOLUME = 1.15
MIN_SECONDS_BETWEEN_SCENARIO_SFX = 1.25
PRE_HIGHLIGHT_SFX_OFFSET = 0.10
ADVANCED_SPEAKER_TRACKING = True
FACE_TRACK_STEP_SECONDS = 0.25
FACE_TRACK_SMOOTHING = 0.30
FACE_TRACK_SNAP_RATIO = 0.18
FACE_TRACK_DEADZONE_RATIO = 0.12
FACE_TRACK_MIN_CONFIDENCE_AREA = 0.0006
HIGHLIGHT_COLOR_WARNING = (236, 29, 29, 255)
HIGHLIGHT_COLOR_KEYTERM = (244, 199, 15, 255)
HIGHLIGHT_COLOR_EMOTION = (136, 255, 196, 255)
HIGHLIGHT_COLOR_DOMAIN = (153, 223, 0, 255)
HIGHLIGHT_COLORS = [
    HIGHLIGHT_COLOR_WARNING,
    HIGHLIGHT_COLOR_KEYTERM,
    HIGHLIGHT_COLOR_EMOTION,
    HIGHLIGHT_COLOR_DOMAIN,
]
HIGHLIGHT_STOPWORDS = {
    "aber", "alle", "alles", "als", "also", "am", "an", "auch", "auf", "aus", "bei",
    "bin", "bis", "da", "dann", "das", "dass", "dein", "deine", "dem", "den", "der",
    "des", "die", "dies", "diese", "doch", "du", "ein", "eine", "einem", "einen",
    "einer", "es", "fuer", "ganz", "hab", "habe", "haben", "hat", "hatte", "hier",
    "genau", "gerne", "glaube", "ich", "ihm", "ihn", "im", "immer", "in", "ist", "ja", "jetzt", "kann", "mal",
    "man", "mein", "meine", "mit", "noch", "oder", "ohne", "schon", "sehr", "sein",
    "sich", "sie", "sind", "so", "und", "uns", "unser", "unsere", "vom", "von",
    "war", "was", "wenn", "wer", "wie", "wir", "wo", "zu", "zum", "zur",
    "dich", "dir", "mir", "mich", "neben", "erstmal", "spaeter", "später", "heute",
}
PERSON_NAME_STOPWORDS = {
    "max", "alex", "anna", "ben", "felix", "jan", "julia", "lena", "leon", "lisa",
    "maria", "marie", "markus", "moritz", "paul", "sara", "sarah", "sophie", "tom",
}
COMPANY_INSTITUTION_TERMS = {
    "hhl", "hochschule", "universitaet", "university", "uni", "businessschool", "businessschools",
    "school", "schools", "roland", "berger", "unternehmen", "firma", "institut", "fakultaet",
    "lehrstuhl", "campus",
}
DOMAIN_TERMS = {
    "finance", "master", "masterprogramm", "masterprogramme", "business", "strategieberatung",
    "beratung", "karriere", "berufserfahrung", "studierende", "studierenden", "programme",
    "programm", "netzwerk", "community", "unternehmerisch", "unternehmerische", "denken",
    "handeln", "moeglichkeiten", "netzwerk",
}
EMOTIONAL_TERMS = {
    "persoenlich", "persoenliche", "persoenlicher", "interaktiv", "begeistern", "antreiben",
    "passion", "vereint", "eng", "beherzt", "besonders", "auszeichnet", "community",
    "zusammenhalt", "motivation", "chance", "chancen",
}
WARNING_TERMS = {
    "problem", "probleme", "stress", "druck", "angst", "risiko", "riskant", "fehler",
    "mythos", "mythen", "falsch", "deadline", "krise",
}


def _normalize_token(token):
    token = token.lower()
    token = (
        token
        .replace("\u00e4", "ae")
        .replace("\u00f6", "oe")
        .replace("\u00fc", "ue")
        .replace("\u00df", "ss")
    )
    return re.sub(r"[^a-z0-9]+", "", token)


def _load_emoji_mapping():
    if not os.path.exists(EMOJI_MAPPING_PATH):
        return {}
    with open(EMOJI_MAPPING_PATH, "r", encoding="utf-8") as mapping_file:
        return json.load(mapping_file)


def _emoji_lookup_from_mapping():
    lookup = {}
    for image_name, words in _load_emoji_mapping().items():
        for word in words:
            normalized = _normalize_token(word)
            if normalized:
                lookup[normalized] = image_name
    return lookup


def _highlight_tokens(highlight_words):
    tokens = set()
    for phrase in highlight_words or []:
        for token in str(phrase).split():
            normalized = _normalize_token(token)
            if normalized:
                tokens.add(normalized)
    return tokens


def _highlight_token_colors(highlight_words, start_index):
    token_colors = {}
    color_index = start_index
    for phrase in highlight_words or []:
        phrase_tokens = []
        for token in str(phrase).split():
            normalized = _normalize_token(token)
            if normalized:
                phrase_tokens.append(normalized)
        if not phrase_tokens:
            continue
        color = HIGHLIGHT_COLORS[color_index % len(HIGHLIGHT_COLORS)]
        color_index += 1
        for token in phrase_tokens:
            token_colors[token] = color
    return token_colors, color_index


def _clip_highlights_by_time(editing_script):
    highlights = []
    color_index = 0
    for clip_info in editing_script.get("clips", []):
        start = float(clip_info.get("start", 0.0))
        end = float(clip_info.get("end", start))
        highlight_words = clip_info.get("highlight_words", [])
        tokens = _highlight_tokens(highlight_words)
        if end > start and tokens:
            token_colors, color_index = _highlight_token_colors(highlight_words, color_index)
            highlights.append((start, end, tokens, token_colors))
    return highlights


def _highlight_color_for_chunk(chunk_words, start, end, highlight_windows):
    return CAPTION_COLOR


def _highlight_color_for_word(word, start, end, highlight_windows):
    normalized = _normalize_token(word)
    for highlight_start, highlight_end, highlight_tokens, token_colors in highlight_windows:
        overlaps = start < highlight_end and end > highlight_start
        if overlaps and normalized in highlight_tokens:
            return token_colors.get(normalized, HIGHLIGHT_COLORS[0])
    return CAPTION_COLOR


def _word_text(word_info):
    if isinstance(word_info, dict):
        return str(word_info.get("word", "")).strip()
    return str(word_info).strip()


def _display_word(word):
    word = str(word).strip()
    return word[:MAX_CAPTION_CHARS]


def _caption_text_for_words(words):
    lines = _caption_lines_for_words(words)
    return "\n".join(" ".join(line) for line in lines)


def _is_long_word(word):
    return len(_normalize_token(word)) >= LONG_WORD_MIN_CHARS


def _caption_lines_for_words(words):
    lines = []
    current_line = []

    for word_info in words:
        word = _display_word(_word_text(word_info))
        if not word:
            continue

        if _is_long_word(word):
            if current_line:
                lines.append(current_line)
                current_line = []
            lines.append([word])
            continue

        if len(current_line) >= MAX_SHORT_WORDS_PER_LINE or any(_is_long_word(existing) for existing in current_line):
            lines.append(current_line)
            current_line = [word]
        else:
            current_line.append(word)

    if current_line:
        lines.append(current_line)

    return lines


def _fits_caption_limits(words):
    text = _caption_text_for_words(words)
    if len(text.replace("\n", " ")) > MAX_CAPTION_CHARS:
        return False

    lines = _caption_lines_for_words(words)
    if len(lines) > MAX_LINES_PER_CAPTION:
        return False

    for line in lines:
        if any(_is_long_word(word) for word in line):
            if len(line) != 1:
                return False
        elif len(line) > MAX_SHORT_WORDS_PER_LINE:
            return False

    return True


def _chunk_duration_with(candidate):
    try:
        return float(candidate[-1]["end"]) - float(candidate[0]["start"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _should_start_new_chunk(candidate):
    if len(candidate) <= 1:
        return False
    first_block = candidate[0].get("block_index")
    last_block = candidate[-1].get("block_index")
    if first_block is not None and last_block is not None and first_block != last_block:
        return True
    if not _fits_caption_limits(candidate):
        return True
    return _chunk_duration_with(candidate) > MAX_CAPTION_DURATION


def _should_end_chunk(word_info, chunk):
    word = _word_text(word_info)
    if not _fits_caption_limits(chunk):
        return True
    if len(chunk) >= MIN_WORDS_PER_CAPTION and word.endswith((".", "!", "?", ",", ";", ":")):
        return True
    return False


def _caption_events_from_transcript(transcript, editing_script):
    highlight_windows = _clip_highlights_by_time(editing_script)
    question_blocks = _question_block_indices(transcript)
    events = []
    chunk = []

    for word_info in transcript or []:
        word = str(word_info.get("word", "")).strip()
        if not word:
            continue

        candidate = [*chunk, word_info]
        if chunk and _should_start_new_chunk(candidate):
            events.append(_event_from_chunk(chunk, highlight_windows, question_blocks))
            chunk = [word_info]
        else:
            chunk = candidate

        if _should_end_chunk(word_info, chunk):
            events.append(_event_from_chunk(chunk, highlight_windows, question_blocks))
            chunk = []

    if chunk:
        events.append(_event_from_chunk(chunk, highlight_windows, question_blocks))

    return events


def _question_block_indices(transcript):
    block_words = {}
    for word_info in transcript or []:
        block_index = word_info.get("block_index")
        if block_index is None:
            continue
        block_words.setdefault(block_index, []).append(str(word_info.get("word", "")).strip())

    question_blocks = set()
    for block_index, words in block_words.items():
        text = " ".join(word for word in words if word)
        normalized = " ".join(_normalize_token(word) for word in words)
        starts_like_question = normalized.startswith((
            "erzaehl",
            "also was",
            "die hhl",
            "du blickst",
            "was",
            "wer",
            "wie",
            "warum",
            "wieso",
        ))
        if text.endswith("?") or starts_like_question:
            question_blocks.add(block_index)
    return question_blocks


def _auto_emoji_infos_from_caption_events(caption_events):
    emoji_lookup = _emoji_lookup_from_mapping()
    if not emoji_lookup:
        return []

    emoji_infos = []
    last_emoji_start = -999.0
    for event in caption_events:
        ordered_tokens = [_normalize_token(word) for word in event.get("words", [])]
        ordered_tokens = [token for token in ordered_tokens if token]
        priority_tokens = [token for token in ordered_tokens if token in THINKING_EMOJI_TOKENS]
        if not priority_tokens and event["start"] - last_emoji_start < MIN_SECONDS_BETWEEN_EMOJIS:
            continue

        for token in [*priority_tokens, *ordered_tokens]:
            image_name = emoji_lookup.get(token)
            if not image_name:
                continue
            if token in THINKING_EMOJI_TOKENS:
                image_name = THINKING_EMOJI_NAME

            emoji_path = os.path.join(EMOJI_ASSET_DIR, image_name)
            if not os.path.exists(emoji_path):
                continue

            emoji_infos.append({
                "image": image_name,
                "start": event["start"],
                "end": event["end"],
                "matched_word": token,
            })
            last_emoji_start = event["start"]
            break

    return emoji_infos


def _is_priority_emoji_info(emoji_info):
    image_name = os.path.basename(emoji_info.get("image", "")).lower()
    matched_word = _normalize_token(emoji_info.get("matched_word", ""))
    return image_name == THINKING_EMOJI_NAME and matched_word in THINKING_EMOJI_TOKENS


def _emoji_start_block_seconds(original_video_path):
    video_name = os.path.basename(original_video_path).lower()
    return NO_EMOJI_START_SECONDS_BY_VIDEO.get(video_name, 0.0)


def _event_from_chunk(chunk_words, highlight_windows, question_blocks=None):
    start = float(chunk_words[0]["start"])
    end = float(chunk_words[-1]["end"])
    if end <= start:
        end = start + 0.25

    text = _caption_text_for_words(chunk_words)
    question_blocks = question_blocks or set()
    chunk_blocks = {
        word.get("block_index")
        for word in chunk_words
        if word.get("block_index") is not None
    }
    is_question = (
        any(str(word.get("word", "")).strip().endswith("?") for word in chunk_words)
        or bool(chunk_blocks & question_blocks)
    )

    return {
        "text": text,
        "start": start,
        "end": end,
        "color": _highlight_color_for_chunk(chunk_words, start, end, highlight_windows),
        "words": [_display_word(word["word"]) for word in chunk_words],
        "word_starts": [float(word.get("start", start)) for word in chunk_words],
        "word_ends": [float(word.get("end", end)) for word in chunk_words],
        "word_colors": [CAPTION_COLOR for word in chunk_words],
        "tokens": {_normalize_token(word["word"]) for word in chunk_words},
        "is_question": is_question,
        "block_start": chunk_words[0].get("block_index"),
        "block_end": chunk_words[-1].get("block_index"),
        "speakers": [word.get("speaker") for word in chunk_words if word.get("speaker")],
    }


def _is_noun_candidate(word):
    normalized = _normalize_token(word)
    raw_word = str(word).strip()
    if len(normalized) < 3 or normalized in HIGHLIGHT_STOPWORDS:
        return False
    return raw_word[:1].isupper()


def _is_highlight_candidate(word):
    return _is_noun_candidate(word)


def _select_highlight_word_index(event, highlight_windows):
    words = event.get("words") or [word for line in event["text"].splitlines() for word in line.split()]
    for index, word in enumerate(words):
        normalized = _normalize_token(word)
        for highlight_start, highlight_end, highlight_tokens, _token_colors in highlight_windows:
            overlaps = event["start"] < highlight_end and event["end"] > highlight_start
            if overlaps and normalized in highlight_tokens:
                return index

    for index, word in enumerate(words):
        if _is_highlight_candidate(word):
            return index
    return None


def _is_acronym(word):
    raw_word = re.sub(r"[^A-Za-zÄÖÜäöüß]+", "", str(word).strip())
    return len(raw_word) >= 2 and raw_word.upper() == raw_word


def _highlight_category_for_token(word, normalized, explicit_highlight):
    if normalized in WARNING_TERMS:
        return "warning"
    if normalized in COMPANY_INSTITUTION_TERMS or _is_acronym(word):
        return "keyterm"
    if normalized in EMOTIONAL_TERMS:
        return "emotion"
    if normalized in DOMAIN_TERMS or explicit_highlight:
        return "domain"
    return None


def _highlight_color_for_category(category):
    if category == "warning":
        return HIGHLIGHT_COLOR_WARNING
    if category == "keyterm":
        return HIGHLIGHT_COLOR_KEYTERM
    if category == "emotion":
        return HIGHLIGHT_COLOR_EMOTION
    return HIGHLIGHT_COLOR_DOMAIN


def _score_highlight_candidate(word, event, highlight_windows, highlighted_terms):
    normalized = _normalize_token(word)
    if not normalized:
        return None
    if normalized in highlighted_terms:
        return None
    if normalized in HIGHLIGHT_STOPWORDS or normalized in PERSON_NAME_STOPWORDS:
        return None

    explicit_highlight = False
    for highlight_start, highlight_end, highlight_tokens, _token_colors in highlight_windows:
        overlaps = event["start"] < highlight_end and event["end"] > highlight_start
        if overlaps and normalized in highlight_tokens:
            explicit_highlight = True
            break

    category = _highlight_category_for_token(word, normalized, explicit_highlight)
    if not category:
        return None

    score = 0
    if category == "keyterm":
        score += 6
    elif category == "domain":
        score += 5
    elif category == "emotion":
        score += 4
    elif category == "warning":
        score += 4

    if explicit_highlight:
        score += 3
    if _is_acronym(word):
        score += 2
    if len(normalized) >= 8:
        score += 1

    return score, category, normalized


def _score_fallback_highlight_candidate(word, highlighted_terms):
    normalized = _normalize_token(word)
    if not normalized:
        return None
    if normalized in highlighted_terms:
        return None
    if normalized in HIGHLIGHT_STOPWORDS or normalized in PERSON_NAME_STOPWORDS:
        return None
    if len(normalized) < 4:
        return None

    score = len(normalized)
    if _is_acronym(word):
        score += 6
    if str(word).strip()[:1].isupper():
        score += 3
    return score, "domain", normalized


def _score_last_resort_highlight_candidate(word, highlighted_terms):
    normalized = _normalize_token(word)
    if not normalized:
        return None
    if normalized in highlighted_terms and len(normalized) < 8:
        return None
    if len(normalized) < 2:
        return None

    score = len(normalized)
    if normalized not in HIGHLIGHT_STOPWORDS:
        score += 5
    if _is_acronym(word):
        score += 6
    if str(word).strip()[:1].isupper():
        score += 2
    return score, "domain", normalized


def _apply_periodic_highlights(caption_events, editing_script):
    highlight_windows = _clip_highlights_by_time(editing_script)
    highlighted_terms = set()
    color_index = 0

    for caption_index, event in enumerate(caption_events):
        words = event.get("words") or [word for line in event["text"].splitlines() for word in line.split()]
        event["word_colors"] = [CAPTION_COLOR for _ in words]
        if (caption_index + 1) % HIGHLIGHT_EVERY_NTH_CAPTION != 0:
            continue

        best_candidate = None
        for index, word in enumerate(words):
            candidate = _score_highlight_candidate(word, event, highlight_windows, highlighted_terms)
            if not candidate:
                continue

            score, category, normalized = candidate
            if not best_candidate or score > best_candidate[0]:
                best_candidate = (score, index, category, normalized)

        if not best_candidate:
            for index, word in enumerate(words):
                candidate = _score_fallback_highlight_candidate(word, highlighted_terms)
                if not candidate:
                    continue

                score, category, normalized = candidate
                if not best_candidate or score > best_candidate[0]:
                    best_candidate = (score, index, category, normalized)

        if not best_candidate:
            for index, word in enumerate(words):
                candidate = _score_last_resort_highlight_candidate(word, highlighted_terms)
                if not candidate:
                    continue

                score, category, normalized = candidate
                if not best_candidate or score > best_candidate[0]:
                    best_candidate = (score, index, category, normalized)

        if not best_candidate:
            continue

        _score, selected_index, category, normalized = best_candidate
        event["word_colors"][selected_index] = HIGHLIGHT_COLORS[color_index % len(HIGHLIGHT_COLORS)]
        color_index += 1


def _caption_events_from_script(editing_script):
    events = []
    highlight_windows = _clip_highlights_by_time(editing_script)
    for clip_info in editing_script.get("clips", []):
        words = [word for word in str(clip_info.get("caption_text", "")).split() if word]
        if not words:
            continue

        start = float(clip_info.get("start", 0.0))
        end = float(clip_info.get("end", start))
        if end <= start:
            continue

        chunks = _split_words_into_chunks(words)
        duration = (end - start) / len(chunks)
        for index, chunk in enumerate(chunks):
            chunk_start = start + index * duration
            chunk_end = end if index == len(chunks) - 1 else start + (index + 1) * duration
            pseudo_words = [{"word": word, "start": chunk_start, "end": chunk_end} for word in chunk]
            events.append(_event_from_chunk(pseudo_words, highlight_windows))

    return events


def _load_pil_font():
    if os.path.exists(FONT_BLACK_PATH):
        return ImageFont.truetype(FONT_BLACK_PATH, CAPTION_FONT_SIZE)
    if os.path.exists(FONT_BOLD_PATH):
        return ImageFont.truetype(FONT_BOLD_PATH, CAPTION_FONT_SIZE)
    if os.path.exists(FONT_REGULAR_PATH):
        return ImageFont.truetype(FONT_REGULAR_PATH, CAPTION_FONT_SIZE)
    return ImageFont.load_default()


def _split_words_into_chunks(words):
    chunks = []
    current = []
    for word in words:
        candidate = [*current, word]
        fake_word_info = {"word": word}
        if current and _should_start_new_chunk([{"word": chunk_word, "start": 0, "end": 0} for chunk_word in candidate]):
            chunks.append(current)
            current = [word]
        else:
            current = candidate
        if _should_end_chunk(fake_word_info, current):
            chunks.append(current)
            current = []
    if current:
        if chunks and len(current) < MIN_WORDS_PER_CAPTION and _fits_caption_limits([*chunks[-1], *current]):
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


def _text_bbox(text, font):
    scratch = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(scratch)
    return draw.textbbox((0, 0), text, font=font, stroke_width=CAPTION_STROKE_WIDTH)


def _make_pil_text_clip(text, color, word_colors=None):
    font = _load_pil_font()
    lines = [line.split() for line in text.splitlines() if line.strip()]
    words = [word for line in lines for word in line]
    if not lines:
        lines = [[text]]
        words = [text]

    if not word_colors or len(word_colors) != len(words):
        word_colors = [color] * len(words)

    space_width = int(round(font.getlength(" ")))
    line_word_bboxes = [[_text_bbox(word, font) for word in line] for line in lines]
    line_widths = [
        sum(bbox[2] - bbox[0] for bbox in bboxes) + space_width * max(0, len(bboxes) - 1)
        for bboxes in line_word_bboxes
    ]
    line_heights = [max(bbox[3] - bbox[1] for bbox in bboxes) for bboxes in line_word_bboxes]
    line_tops = [min(bbox[1] for bbox in bboxes) for bboxes in line_word_bboxes]
    line_spacing = int(CAPTION_FONT_SIZE * 0.12)
    width = max(line_widths) + CAPTION_STROKE_WIDTH * 4
    height = sum(line_heights) + line_spacing * max(0, len(lines) - 1) + CAPTION_STROKE_WIDTH * 4
    image = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    color_cursor = 0
    y = CAPTION_STROKE_WIDTH * 2
    for line, bboxes, line_width, line_height, line_top in zip(lines, line_word_bboxes, line_widths, line_heights, line_tops):
        x = (width - line_width) / 2
        baseline_y = y - line_top
        for word, bbox in zip(line, bboxes):
            draw.text(
                (x - bbox[0], baseline_y),
                word,
                font=font,
                fill=word_colors[color_cursor],
                stroke_width=CAPTION_STROKE_WIDTH,
                stroke_fill=CAPTION_STROKE_COLOR,
            )
            x += bbox[2] - bbox[0] + space_width
            color_cursor += 1
        y += line_height + line_spacing
    return ImageClip(np.array(image))


def _make_text_clip(text, color, word_colors=None, max_width=None):
    text_clip = _make_pil_text_clip(text, color, word_colors)
    if max_width and text_clip.w > max_width:
        text_clip = text_clip.resize(width=max_width)
    return text_clip


def _position_centered_at_y(clip, base_height, y_ratio):
    return ("center", int(base_height * y_ratio - clip.h / 2))


def _make_caption_clip(event, base_clip):
    duration = max(0.05, event["end"] - event["start"])
    safe_width = int(base_clip.w * CAPTION_SAFE_WIDTH_RATIO)
    text_clip = _make_text_clip(event["text"], event["color"], event.get("word_colors"), safe_width)
    position = _position_centered_at_y(text_clip, base_clip.h, CAPTION_Y_RATIO)
    event["caption_y"] = int(position[1])
    event["caption_h"] = int(text_clip.h)
    return (
        text_clip
        .set_start(event["start"])
        .set_duration(duration)
        .set_position(position)
    )


def _find_caption_event_for_emoji(emoji_info, caption_events):
    matched_word = _normalize_token(emoji_info.get("matched_word", ""))
    emoji_start = float(emoji_info.get("start", 0.0))

    for event in caption_events:
        if matched_word and matched_word in event["tokens"]:
            return event

    for event in caption_events:
        if event["start"] <= emoji_start < event["end"]:
            return event

    return None


def _caption_event_key(event):
    return (round(float(event["start"]), 2), round(float(event["end"]), 2), event["text"])


def _make_emoji_clip(emoji_info, caption_events, base_clip, emoji_start_block_seconds=0.0, target_event=None):
    image_name = os.path.basename(emoji_info.get("image", ""))
    if not image_name:
        return None

    if float(emoji_info.get("start", 0.0)) < emoji_start_block_seconds:
        return None

    emoji_path = os.path.join(EMOJI_ASSET_DIR, image_name)
    if not os.path.exists(emoji_path):
        logging.warning("Emoji asset not found: %s", emoji_path)
        return None

    target_event = target_event or _find_caption_event_for_emoji(emoji_info, caption_events)
    if not target_event:
        return None

    emoji_size = HANDSHAKE_EMOJI_SIZE if image_name.lower() == HANDSHAKE_EMOJI_NAME else (EMOJI_SIZE, EMOJI_SIZE)
    with Image.open(emoji_path).convert("RGBA") as emoji_image:
        emoji_image = emoji_image.resize(emoji_size, Image.Resampling.LANCZOS)
        emoji_clip = ImageClip(np.array(emoji_image))
    duration = max(0.05, target_event["end"] - target_event["start"])
    emoji_y = target_event.get("caption_y", int(base_clip.h * CAPTION_Y_RATIO)) + target_event.get("caption_h", 0) + EMOJI_GAP_PX
    emoji_y = max(0, min(int(emoji_y), int(base_clip.h - emoji_size[1])))
    return (
        emoji_clip
        .set_start(target_event["start"])
        .set_duration(duration)
        .set_position(("center", emoji_y))
    )


def _all_emoji_infos(editing_script):
    emojis = []
    for clip_info in editing_script.get("clips", []):
        emojis.extend(clip_info.get("emojis", []))
    return emojis


def _importance_zoom_events(editing_script, caption_events):
    events = _answer_start_zoom_events_from_captions(caption_events)
    if not events:
        events = _answer_start_zoom_events(editing_script, caption_events)
    if events:
        logging.info("Prepared %s answer punch-in zoom events.", len(events))
    return events


def _answer_start_zoom_events_from_captions(caption_events):
    events = []
    last_start = -999.0
    pending_question_end = None
    ordered_events = sorted(caption_events or [], key=lambda item: float(item.get("start", 0.0)))
    usable_speaker_labels = len({
        speaker
        for event in ordered_events
        for speaker in (event.get("speakers") or [])
        if speaker
    }) >= 2

    for index, event in enumerate(ordered_events):
        start = float(event.get("start", 0.0))
        end = float(event.get("end", start))
        if event.get("is_question"):
            pending_question_end = end
            continue

        if pending_question_end is None:
            continue
        if start - pending_question_end > 4.5:
            pending_question_end = None
            continue
        if start - last_start < MIN_SECONDS_BETWEEN_IMPORTANCE_ZOOMS:
            pending_question_end = None
            continue

        preliminary_end = _next_question_start_or_timeline_end(ordered_events, index)
        answer_speaker = (
            _dominant_speaker_for_caption_events(ordered_events[index:], start, preliminary_end)
            if usable_speaker_labels
            else None
        )
        answer_end = _answer_end_for_active_speaker(ordered_events, index, answer_speaker)
        if answer_end <= start:
            pending_question_end = None
            continue

        events.append({
            "start": start,
            "end": answer_end,
            "scale": IMPORTANCE_ZOOM_SCALE,
            "speaker": answer_speaker,
        })
        last_start = start
        pending_question_end = None

    return events


def _next_question_start_or_timeline_end(caption_events, start_index):
    timeline_end = float(caption_events[start_index].get("end", caption_events[start_index].get("start", 0.0)))
    for event in caption_events[start_index + 1:]:
        event_start = float(event.get("start", 0.0))
        event_end = float(event.get("end", event_start))
        if event.get("is_question"):
            return event_start
        timeline_end = max(timeline_end, event_end)
    return timeline_end


def _dominant_speaker_for_event(event):
    speakers = event.get("speakers") or []
    if not speakers:
        return None
    return max(set(speakers), key=speakers.count)


def _answer_end_for_active_speaker(caption_events, start_index, answer_speaker):
    start_event = caption_events[start_index]
    answer_start = float(start_event.get("start", 0.0))
    timeline_end = float(start_event.get("end", answer_start))

    for event in caption_events[start_index + 1:]:
        event_start = float(event.get("start", 0.0))
        event_end = float(event.get("end", event_start))
        if event.get("is_question"):
            return event_start

        event_speaker = _dominant_speaker_for_event(event)
        if answer_speaker and event_speaker and event_speaker != answer_speaker and event_start - answer_start >= 0.65:
            return event_start

        timeline_end = max(timeline_end, event_end)

    return timeline_end


def _answer_start_zoom_events(editing_script, caption_events):
    events = []
    last_start = -999.0
    clips = sorted(editing_script.get("clips", []), key=lambda item: float(item.get("start", 0.0)))

    for index, clip_info in enumerate(clips):
        if not clip_info.get("is_question"):
            continue

        question_end = float(clip_info.get("end", clip_info.get("start", 0.0)))
        answer_clip = None
        for candidate in clips[index + 1:]:
            candidate_start = float(candidate.get("start", 0.0))
            candidate_end = float(candidate.get("end", candidate_start))
            if candidate_end <= question_end:
                continue
            if candidate_start - question_end > 4.0:
                break
            if not candidate.get("is_question"):
                answer_clip = candidate
                break

        if not answer_clip:
            continue

        answer_start = float(answer_clip.get("start", question_end))
        answer_end = float(answer_clip.get("end", answer_start))
        target_event = _first_caption_event_in_window(caption_events, answer_start, answer_end)
        if target_event:
            answer_start = max(answer_start, float(target_event.get("start", answer_start)))

        if answer_start - last_start < MIN_SECONDS_BETWEEN_IMPORTANCE_ZOOMS:
            continue

        duration = max(IMPORTANCE_ZOOM_MIN_DURATION, answer_end - answer_start)
        events.append({
            "start": answer_start,
            "end": min(answer_start + duration, answer_end),
            "scale": IMPORTANCE_ZOOM_SCALE,
            "speaker": _dominant_speaker_for_caption_events(caption_events, answer_start, answer_end),
        })
        last_start = answer_start

    return events


def _dominant_speaker_for_caption_events(caption_events, start, end):
    speakers = []
    for event in caption_events or []:
        event_start = float(event.get("start", 0.0))
        event_end = float(event.get("end", event_start))
        if event_start >= end or event_end <= start:
            continue
        speakers.extend(event.get("speakers") or [])
    if not speakers:
        return None
    return max(set(speakers), key=speakers.count)


def _first_caption_event_in_window(caption_events, start, end):
    for event in caption_events:
        event_start = float(event.get("start", 0.0))
        if start - 0.05 <= event_start < end:
            return event
    return None


def _clip_for_time(clips, time_seconds):
    for clip in clips or []:
        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", start))
        if start <= time_seconds < end:
            return clip
    return None


def _ease_out_cubic(value):
    value = max(0.0, min(1.0, value))
    return 1 - (1 - value) ** 3


def _zoom_event_at_time(time_seconds, zoom_events):
    for event in zoom_events:
        start = float(event.get("start", 0.0)) if isinstance(event, dict) else float(event[0])
        end = float(event.get("end", start)) if isinstance(event, dict) else float(event[1])
        scale = float(event.get("scale", IMPORTANCE_ZOOM_SCALE)) if isinstance(event, dict) else float(event[2])
        if start <= time_seconds <= end:
            return {**event, "scale": scale} if isinstance(event, dict) else {"start": start, "end": end, "scale": scale}
        if end < time_seconds <= end + 0.12:
            progress = (time_seconds - end) / 0.12
            eased_scale = scale + (1.0 - scale) * min(1.0, progress * progress)
            return {**event, "scale": eased_scale} if isinstance(event, dict) else {"start": start, "end": end, "scale": eased_scale}
    return None


def _apply_importance_zooms(base_clip, zoom_events):
    if not zoom_events:
        return base_clip

    def zoom_frame(get_frame, time_seconds):
        zoom_event = _zoom_event_at_time(time_seconds, zoom_events)
        scale = float(zoom_event.get("scale", 1.0)) if zoom_event else 1.0
        if scale <= 1.001:
            return get_frame(time_seconds)

        frame = get_frame(time_seconds)
        height, width = frame.shape[:2]
        crop_width = max(1, int(width / scale))
        crop_height = max(1, int(height / scale))
        center_ratio = float(zoom_event.get("center_x_ratio", 0.5))
        center_x = width * max(0.0, min(1.0, center_ratio))
        center_y_ratio = float(zoom_event.get("center_y_ratio", ANSWER_ZOOM_Y_CENTER_RATIO))
        center_y = height * max(0.0, min(1.0, center_y_ratio))
        x1 = int(max(0, min(width - crop_width, center_x - crop_width / 2)))
        y1 = int(max(0, min(height - crop_height, center_y - crop_height / 2)))
        cropped = frame[y1:y1 + crop_height, x1:x1 + crop_width]

        image = Image.fromarray(cropped)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(image)

    logging.info("Applying %s importance zooms.", len(zoom_events))
    return base_clip.fl(zoom_frame)


def _importance_zoom_sfx_clips(assets, zoom_events, final_duration):
    sfx_path = (assets.get("sfx") or {}).get("importance_zoom")
    if not sfx_path or not os.path.exists(sfx_path):
        return []

    clips = []
    for event in zoom_events:
        start = float(event.get("start", 0.0)) if isinstance(event, dict) else float(event[0])
        end = float(event.get("end", start)) if isinstance(event, dict) else float(event[1])
        if start >= final_duration:
            continue
        try:
            sfx_clip = AudioFileClip(sfx_path).fx(audio_normalize).volumex(IMPORTANCE_ZOOM_SFX_VOLUME)
            max_duration = max(0.05, min(end - start, final_duration - start, sfx_clip.duration))
            clips.append(sfx_clip.subclip(0, max_duration).set_start(start))
        except Exception as exc:
            logging.warning("Could not add importance zoom SFX: %s", exc)
    return clips


def _opening_hit_sfx_clip(assets, final_duration):
    sfx_path = (assets.get("sfx") or {}).get("opening_hit")
    if not sfx_path or not os.path.exists(sfx_path) or final_duration <= 0:
        return None

    try:
        sfx_clip = AudioFileClip(sfx_path).fx(audio_normalize).volumex(OPENING_HIT_SFX_VOLUME)
        max_duration = min(0.75, final_duration, sfx_clip.duration)
        return sfx_clip.subclip(0, max_duration).set_start(0.05)
    except Exception as exc:
        logging.warning("Could not add opening hit SFX: %s", exc)
        return None


def _make_sfx_clip(sfx_path, start, final_duration, volume, max_clip_duration=0.85):
    if not sfx_path or not os.path.exists(sfx_path) or start >= final_duration:
        return None
    try:
        sfx_clip = AudioFileClip(sfx_path).fx(audio_normalize).volumex(volume)
        duration = min(max_clip_duration, final_duration - start, sfx_clip.duration)
        return sfx_clip.subclip(0, max(0.05, duration)).set_start(max(0.0, start))
    except Exception as exc:
        logging.warning("Could not add scenario SFX %s: %s", sfx_path, exc)
        return None


def _scenario_sfx_events(editing_script, caption_events, final_duration):
    events = []
    last_sfx_start = -999.0
    answer_clips = [
        clip for clip in editing_script.get("clips", [])
        if not clip.get("is_question") and float(clip.get("end", 0)) > float(clip.get("start", 0))
    ]

    for index, clip in enumerate(answer_clips):
        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", start))
        if start >= final_duration:
            continue

        if index > 0 and start - last_sfx_start >= MIN_SECONDS_BETWEEN_SCENARIO_SFX * 1.6:
            sfx_start = max(0.0, start - PRE_HIGHLIGHT_SFX_OFFSET)
            events.append(("topic_wosh", sfx_start, TOPIC_WOSH_SFX_VOLUME, 1.25))
            last_sfx_start = sfx_start

        if (clip.get("importance_score", 0) >= 6 or clip.get("is_enumeration")) and end - last_sfx_start >= MIN_SECONDS_BETWEEN_SCENARIO_SFX * 1.8:
            events.append(("topic_riser", max(start, end - 1.40), TOPIC_RISER_SFX_VOLUME, 1.40))
            last_sfx_start = end

    last_wosh_start = -999.0
    for event in caption_events:
        start = float(event.get("start", 0.0))
        if start < 12.0 or start >= final_duration - 4.0:
            continue
        if event.get("is_question"):
            continue
        has_highlight = any(color != CAPTION_COLOR for color in event.get("word_colors", []))
        if not has_highlight:
            continue
        if start - last_wosh_start < 17.0:
            continue
        wosh_start = max(0.0, start - PRE_HIGHLIGHT_SFX_OFFSET)
        events = [
            item for item in events
            if not (item[0] == "keyword_ping" and abs(float(item[1]) - wosh_start) < 0.55)
        ]
        events.append(("topic_wosh", wosh_start, TOPIC_WOSH_SFX_VOLUME, 1.25))
        last_wosh_start = start

    return sorted(events, key=lambda item: item[1])


def _has_nearby_sfx(events, target_start, min_gap):
    return any(abs(float(start) - target_start) < min_gap for _name, start, _volume, _duration in events)


def _sfx_ping_targets_for_clip(clip, caption_events, highlight_terms):
    targets = []
    clip_start = float(clip.get("start", 0.0))
    clip_end = float(clip.get("end", clip_start))
    highlight_tokens = {_normalize_token(term) for term in highlight_terms}

    for event in caption_events:
        if event["start"] < clip_start or event["start"] >= clip_end:
            continue
        if event.get("is_question"):
            continue
        tokens = event.get("tokens", set())
        if highlight_tokens and tokens & highlight_tokens:
            targets.extend(_precise_highlight_sfx_targets(event, highlight_tokens))
        elif clip.get("is_enumeration") and len(event.get("words", [])) <= 2:
            targets.append(max(0.0, float(event["start"]) - PRE_HIGHLIGHT_SFX_OFFSET))
        elif any(color != CAPTION_COLOR for color in event.get("word_colors", [])):
            targets.extend(_precise_highlight_sfx_targets(event, set()))

    if not targets and highlight_terms:
        targets.append(max(0.0, clip_start + 0.12 - PRE_HIGHLIGHT_SFX_OFFSET))
    return sorted(set(round(target, 3) for target in targets))[:3]


def _precise_highlight_sfx_targets(event, highlight_tokens):
    targets = []
    words = event.get("words") or []
    word_starts = event.get("word_starts") or []
    word_colors = event.get("word_colors") or []
    for index, word in enumerate(words):
        normalized = _normalize_token(word)
        color_hit = index < len(word_colors) and word_colors[index] != CAPTION_COLOR
        token_hit = bool(highlight_tokens and normalized in highlight_tokens)
        if not color_hit and not token_hit:
            continue
        if index < len(word_starts):
            targets.append(max(0.0, float(word_starts[index]) - PRE_HIGHLIGHT_SFX_OFFSET))
    return targets


def _scenario_sfx_clips(assets, editing_script, caption_events, final_duration):
    sfx_assets = assets.get("sfx") or {}
    clips = []
    for name, start, volume, max_duration in _scenario_sfx_events(editing_script, caption_events, final_duration):
        clip = _make_sfx_clip(sfx_assets.get(name), start, final_duration, volume, max_duration)
        if clip:
            clips.append(clip)
    logging.info("Added %s scenario SFX clips.", len(clips))
    return clips


def _active_speaker_at(transcript, time_seconds):
    speakers = []
    for word in transcript or []:
        speaker = word.get("speaker")
        if not speaker:
            continue
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        if start - 0.15 <= time_seconds <= end + 0.25:
            speakers.append(speaker)
    if not speakers:
        return None
    return max(set(speakers), key=speakers.count)


def _detect_face_track_samples(video_path, duration):
    try:
        import cv2
    except ImportError:
        logging.info("OpenCV is not installed; speaker tracking is skipped.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples = []
    current_time = 0.0

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
        face_infos = []
        for x, y, w, h in faces:
            area_ratio = (w * h) / max(1, width * height)
            if area_ratio < FACE_TRACK_MIN_CONFIDENCE_AREA:
                continue
            face_infos.append({
                "cx": x + w / 2,
                "cy": y + h / 2,
                "area": w * h,
                "box": (x, y, x + w, y + h),
            })
        face_infos.sort(key=lambda face: face["cx"])
        samples.append({"time": current_time, "faces": face_infos})
        current_time += FACE_TRACK_STEP_SECONDS

    cap.release()
    return samples


def _speaker_center_profiles(face_samples, transcript):
    profiles = {}
    last_by_speaker = {}

    for sample in face_samples:
        faces = sample.get("faces") or []
        if not faces:
            continue

        speaker = _active_speaker_at(transcript, sample["time"])
        if not speaker:
            continue

        if speaker in last_by_speaker:
            target = min(faces, key=lambda face: abs(face["cx"] - last_by_speaker[speaker]))
        else:
            target = max(faces, key=lambda face: face["area"])

        profiles.setdefault(speaker, []).append(target["cx"])
        last_by_speaker[speaker] = target["cx"]

    if not profiles:
        return {}

    import statistics
    return {speaker: statistics.median(values) for speaker, values in profiles.items() if values}


def _face_target_at_time(face_samples, speaker_profiles, transcript, time_seconds, frame_width):
    if not face_samples:
        return None

    times = [sample["time"] for sample in face_samples]
    index = max(0, min(len(face_samples) - 1, bisect.bisect_left(times, time_seconds)))
    sample = face_samples[index]
    faces = sample.get("faces") or []
    if not faces:
        return None

    speaker = _active_speaker_at(transcript, time_seconds)
    if speaker and speaker in speaker_profiles:
        return min(faces, key=lambda face: abs(face["cx"] - speaker_profiles[speaker]))

    return max(faces, key=lambda face: face["area"])


def _face_sample_at_time(face_samples, time_seconds):
    if not face_samples:
        return None
    times = [sample["time"] for sample in face_samples]
    index = max(0, min(len(face_samples) - 1, bisect.bisect_left(times, time_seconds)))
    return face_samples[index]


def _face_target_for_zoom_event(event, face_samples, speaker_profiles, frame_width):
    sample = _face_sample_at_time(face_samples, float(event.get("start", 0.0)))
    faces = (sample or {}).get("faces") or []
    if not faces:
        return None

    speaker = event.get("speaker")
    if speaker and speaker in speaker_profiles:
        return min(faces, key=lambda face: abs(face["cx"] - speaker_profiles[speaker]))

    return max(faces, key=lambda face: face["area"])


def _prepare_answer_focus_zoom_events(zoom_events, face_samples, transcript, frame_width):
    if not zoom_events:
        return []

    distinct_speakers = {word.get("speaker") for word in transcript or [] if word.get("speaker")}
    speaker_profiles = _speaker_center_profiles(face_samples, transcript) if len(distinct_speakers) >= 2 else {}
    prepared = []
    fallback_answerer_center_ratio = None
    for event in zoom_events:
        prepared_event = dict(event) if isinstance(event, dict) else {
            "start": float(event[0]),
            "end": float(event[1]),
            "scale": float(event[2]),
        }
        target = _face_target_for_zoom_event(prepared_event, face_samples, speaker_profiles, frame_width)
        if prepared_event.get("speaker") in speaker_profiles:
            center_ratio = speaker_profiles[prepared_event["speaker"]] / max(1, frame_width)
        elif fallback_answerer_center_ratio is not None:
            center_ratio = fallback_answerer_center_ratio
        elif target:
            center_ratio = target["cx"] / max(1, frame_width)
        else:
            center_ratio = 0.5

        if target or abs(center_ratio - 0.5) > 0.03:
            center_ratio = (
                ANSWER_ZOOM_RIGHT_CENTER_RATIO
                if center_ratio >= 0.5
                else ANSWER_ZOOM_LEFT_CENTER_RATIO
            )

        if not prepared_event.get("speaker") and fallback_answerer_center_ratio is None and target:
            fallback_answerer_center_ratio = center_ratio

        prepared_event["center_x_ratio"] = float(max(0.0, min(1.0, center_ratio)))
        prepared_event["center_y_ratio"] = ANSWER_ZOOM_Y_CENTER_RATIO
        prepared.append(prepared_event)

    side_summary = [
        (
            round(float(event["start"]), 2),
            round(float(event["end"]), 2),
            event.get("speaker"),
            "left" if event.get("center_x_ratio", 0.5) < 0.5 else "right",
        )
        for event in prepared
    ]
    logging.info("Prepared answer focus zoom targets: %s", side_summary)
    return prepared


def _apply_speaker_tracking(base_clip, original_video_path, transcript, face_samples=None):
    if not ADVANCED_SPEAKER_TRACKING:
        return base_clip

    width, height = base_clip.size
    source_ratio = width / max(1, height)
    if source_ratio < 0.70:
        logging.info("Speaker tracking skipped: source is already vertical.")
        return base_clip

    face_samples = face_samples if face_samples is not None else _detect_face_track_samples(original_video_path, base_clip.duration)
    if not face_samples or not any(sample.get("faces") for sample in face_samples):
        logging.info("Speaker tracking skipped: no faces detected.")
        return base_clip

    speaker_profiles = _speaker_center_profiles(face_samples, transcript)
    max_faces = max(len(sample.get("faces") or []) for sample in face_samples)
    if max_faces < 2 and not speaker_profiles:
        logging.info("Speaker tracking skipped: no multi-face or speaker profile signal.")
        return base_clip

    crop_w = int(height * 9 / 16) if source_ratio > 9 / 16 else width
    crop_w = max(2, min(width, crop_w - crop_w % 2))
    crop_h = height
    default_cx = width / 2
    cam_cx = {"value": default_cx}

    def tracking_frame(get_frame, time_seconds):
        frame = get_frame(time_seconds)
        target = _face_target_at_time(face_samples, speaker_profiles, transcript, time_seconds, width)
        target_cx = target["cx"] if target else default_cx

        snap_px = width * FACE_TRACK_SNAP_RATIO
        deadzone_px = crop_w * FACE_TRACK_DEADZONE_RATIO
        current_cx = cam_cx["value"]
        if abs(target_cx - current_cx) > snap_px:
            current_cx = target_cx
        elif target_cx > current_cx + deadzone_px:
            current_cx += (target_cx - (current_cx + deadzone_px)) * FACE_TRACK_SMOOTHING
        elif target_cx < current_cx - deadzone_px:
            current_cx += (target_cx - (current_cx - deadzone_px)) * FACE_TRACK_SMOOTHING

        cam_cx["value"] = current_cx
        x1 = int(max(0, min(width - crop_w, current_cx - crop_w / 2)))
        cropped = frame[0:crop_h, x1:x1 + crop_w]
        image = Image.fromarray(cropped)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(image)

    logging.info("Applying speaker/face tracking with %s samples and %s speaker profiles.", len(face_samples), len(speaker_profiles))
    return base_clip.fl(tracking_frame)


def render_video(original_video_path, editing_script, assets, output_path, transcript=None):
    logging.info("Starting clean overlay rendering process...")
    base_clip = VideoFileClip(original_video_path)
    overlay_clips = []

    caption_events = _caption_events_from_transcript(transcript, editing_script) if transcript else []
    if not caption_events:
        caption_events = _caption_events_from_script(editing_script)
    _apply_periodic_highlights(caption_events, editing_script)
    zoom_events = _importance_zoom_events(editing_script, caption_events)
    face_samples = _detect_face_track_samples(original_video_path, base_clip.duration) if zoom_events or ADVANCED_SPEAKER_TRACKING else []
    zoom_events = _prepare_answer_focus_zoom_events(zoom_events, face_samples, transcript, base_clip.w)
    tracked_base_clip = _apply_speaker_tracking(base_clip, original_video_path, transcript, face_samples)
    visual_base_clip = _apply_importance_zooms(tracked_base_clip, zoom_events)

    for event in caption_events:
        overlay_clips.append(_make_caption_clip(event, visual_base_clip))

    emoji_start_block_seconds = _emoji_start_block_seconds(original_video_path)
    emoji_infos = [*_auto_emoji_infos_from_caption_events(caption_events), *_all_emoji_infos(editing_script)]
    seen_emoji_keys = set()
    seen_emoji_caption_events = set()
    emoji_image_counts = {}
    last_rendered_emoji_start = -999.0
    for emoji_info in emoji_infos:
        emoji_start = float(emoji_info.get("start", 0.0))
        image_name = os.path.basename(emoji_info.get("image", ""))
        is_priority_emoji = _is_priority_emoji_info(emoji_info)
        if emoji_start < emoji_start_block_seconds:
            continue

        if not is_priority_emoji and emoji_start - last_rendered_emoji_start < MIN_SECONDS_BETWEEN_EMOJIS:
            continue

        if emoji_image_counts.get(image_name, 0) >= MAX_SAME_EMOJI_PER_VIDEO:
            continue

        emoji_key = (image_name, round(emoji_start, 2))
        if emoji_key in seen_emoji_keys:
            continue

        target_event = _find_caption_event_for_emoji(emoji_info, caption_events)
        if not target_event:
            continue

        caption_event_key = _caption_event_key(target_event)
        if caption_event_key in seen_emoji_caption_events:
            continue

        emoji_clip = _make_emoji_clip(emoji_info, caption_events, visual_base_clip, emoji_start_block_seconds, target_event)
        if emoji_clip:
            seen_emoji_keys.add(emoji_key)
            seen_emoji_caption_events.add(caption_event_key)
            emoji_image_counts[image_name] = emoji_image_counts.get(image_name, 0) + 1
            last_rendered_emoji_start = emoji_start
            overlay_clips.append(emoji_clip)

    final_video = CompositeVideoClip([visual_base_clip, *overlay_clips], size=base_clip.size)

    all_audio = []
    if base_clip.audio:
        all_audio.append(base_clip.audio.fx(audio_normalize))
    opening_hit = _opening_hit_sfx_clip(assets, final_video.duration)
    if opening_hit:
        all_audio.append(opening_hit)
    all_audio.extend(_scenario_sfx_clips(assets, editing_script, caption_events, final_video.duration))
    if assets.get("music"):
        music_clip = AudioFileClip(assets["music"]).fx(audio_normalize).volumex(0.1)
        if music_clip.duration > final_video.duration:
            music_clip = music_clip.subclip(0, final_video.duration)
        all_audio.append(music_clip)

    if all_audio:
        final_video.audio = CompositeAudioClip(all_audio)

    logging.info("Writing final video to %s...", output_path)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    final_video.close()
    for overlay_clip in overlay_clips:
        overlay_clip.close()
    base_clip.close()
    logging.info("Final video rendering complete!")
