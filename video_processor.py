import logging
import os
import re
import json

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
MIN_SECONDS_BETWEEN_HIGHLIGHTS = 2.5
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
    "ich", "ihm", "ihn", "im", "immer", "in", "ist", "ja", "jetzt", "kann", "mal",
    "man", "mein", "meine", "mit", "noch", "oder", "ohne", "schon", "sehr", "sein",
    "sich", "sie", "sind", "so", "und", "uns", "unser", "unsere", "vom", "von",
    "war", "was", "wenn", "wer", "wie", "wir", "wo", "zu", "zum", "zur",
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
    events = []
    chunk = []

    for word_info in transcript or []:
        word = str(word_info.get("word", "")).strip()
        if not word:
            continue

        candidate = [*chunk, word_info]
        if chunk and _should_start_new_chunk(candidate):
            events.append(_event_from_chunk(chunk, highlight_windows))
            chunk = [word_info]
        else:
            chunk = candidate

        if _should_end_chunk(word_info, chunk):
            events.append(_event_from_chunk(chunk, highlight_windows))
            chunk = []

    if chunk:
        events.append(_event_from_chunk(chunk, highlight_windows))

    return events


def _auto_emoji_infos_from_caption_events(caption_events):
    emoji_lookup = _emoji_lookup_from_mapping()
    if not emoji_lookup:
        return []

    emoji_infos = []
    last_emoji_start = -999.0
    for event in caption_events:
        if event["start"] - last_emoji_start < MIN_SECONDS_BETWEEN_EMOJIS:
            continue

        for token in event["tokens"]:
            image_name = emoji_lookup.get(token)
            if not image_name:
                continue

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


def _emoji_start_block_seconds(original_video_path):
    video_name = os.path.basename(original_video_path).lower()
    return NO_EMOJI_START_SECONDS_BY_VIDEO.get(video_name, 0.0)


def _event_from_chunk(chunk_words, highlight_windows):
    start = float(chunk_words[0]["start"])
    end = float(chunk_words[-1]["end"])
    if end <= start:
        end = start + 0.25

    text = _caption_text_for_words(chunk_words)
    return {
        "text": text,
        "start": start,
        "end": end,
        "color": _highlight_color_for_chunk(chunk_words, start, end, highlight_windows),
        "words": [_display_word(word["word"]) for word in chunk_words],
        "word_colors": [CAPTION_COLOR for word in chunk_words],
        "tokens": {_normalize_token(word["word"]) for word in chunk_words},
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


def _apply_periodic_highlights(caption_events, editing_script):
    highlight_windows = _clip_highlights_by_time(editing_script)
    next_highlight_start = -999.0
    highlighted_terms = set()

    for event in caption_events:
        words = event.get("words") or [word for line in event["text"].splitlines() for word in line.split()]
        event["word_colors"] = [CAPTION_COLOR for _ in words]
        if event["start"] < next_highlight_start:
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
            continue

        _score, selected_index, category, normalized = best_candidate
        event["word_colors"][selected_index] = _highlight_color_for_category(category)
        highlighted_terms.add(normalized)
        next_highlight_start = event["start"] + MIN_SECONDS_BETWEEN_HIGHLIGHTS


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


def render_video(original_video_path, editing_script, assets, output_path, transcript=None):
    logging.info("Starting clean overlay rendering process...")
    base_clip = VideoFileClip(original_video_path)
    overlay_clips = []

    caption_events = _caption_events_from_transcript(transcript, editing_script) if transcript else []
    if not caption_events:
        caption_events = _caption_events_from_script(editing_script)
    _apply_periodic_highlights(caption_events, editing_script)

    for event in caption_events:
        overlay_clips.append(_make_caption_clip(event, base_clip))

    emoji_start_block_seconds = _emoji_start_block_seconds(original_video_path)
    emoji_infos = [*_auto_emoji_infos_from_caption_events(caption_events), *_all_emoji_infos(editing_script)]
    seen_emoji_keys = set()
    seen_emoji_caption_events = set()
    emoji_image_counts = {}
    last_rendered_emoji_start = -999.0
    for emoji_info in emoji_infos:
        emoji_start = float(emoji_info.get("start", 0.0))
        image_name = os.path.basename(emoji_info.get("image", ""))
        if emoji_start < emoji_start_block_seconds:
            continue

        if emoji_start - last_rendered_emoji_start < MIN_SECONDS_BETWEEN_EMOJIS:
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

        emoji_clip = _make_emoji_clip(emoji_info, caption_events, base_clip, emoji_start_block_seconds, target_event)
        if emoji_clip:
            seen_emoji_keys.add(emoji_key)
            seen_emoji_caption_events.add(caption_event_key)
            emoji_image_counts[image_name] = emoji_image_counts.get(image_name, 0) + 1
            last_rendered_emoji_start = emoji_start
            overlay_clips.append(emoji_clip)

    final_video = CompositeVideoClip([base_clip, *overlay_clips], size=base_clip.size)

    all_audio = []
    if base_clip.audio:
        all_audio.append(base_clip.audio.fx(audio_normalize))
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
