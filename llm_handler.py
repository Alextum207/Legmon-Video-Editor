import json
import logging
import os

import litellm
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_MAPPING_PATH = os.path.join(BASE_DIR, "emoji_mapping.json")
EMOJI_ASSET_DIR = r"C:\Users\alexa\OneDrive\Desktop\AI-Video-Editor\assets"
HIGHLIGHT_COLORS = ["#ec1d1d", "#f4c70f", "#88ffc4", "#99df00"]

LITELLM_MODEL = os.getenv("LITELLM_MODEL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")


def load_emoji_mapping():
    with open(EMOJI_MAPPING_PATH, "r", encoding="utf-8") as mapping_file:
        return json.load(mapping_file)


def get_style_guide():
    """Returns a dictionary of all available styles and transitions."""
    return {
        "transitions": [
            {"name": "slide_left", "sfx": "whoosh"},
            {"name": "slide_right", "sfx": "whoosh"},
            {"name": "push_up", "sfx": "whoosh"},
            {"name": "push_down", "sfx": "whoosh"},
            {"name": "flash_frame", "sfx": "light_sfx"},
        ],
        "caption_styles": [
            {"name": "legmon_lustig", "font": "Montserrat Black", "animation_in": "snap_pop", "sfx_in": "pop"},
            {"name": "legmon_reports", "font": "Montserrat Black", "animation_in": "clean_cut", "sfx_in": "tick"},
            {"name": "legmon_interview", "font": "Montserrat Black", "animation_in": "soft_fade", "sfx_in": "whoosh"},
            {"name": "legmon_day_in_a_life", "font": "Montserrat Black", "animation_in": "calm_pop", "sfx_in": "pop"},
        ],
    }


def format_transcript_for_llm(transcript):
    """Formats the word-level transcript into a single string for the LLM prompt."""
    return " ".join(word_info["word"].strip() for word_info in transcript)


def _legmon_few_shots():
    return [
        {
            "style": "Lustig",
            "dynamic": "Fast punchline rhythm, short captions, playful but sparse emojis only on clear mapping matches.",
            "example": {
                "caption_text": "Steuern als Studi? Das klingt erstmal wild.",
                "highlight_words": ["Steuern", "wild"],
                "emojis": [{"image": "money-with-wings.png", "start": 1.0, "end": 2.1, "matched_word": "Steuern"}],
            },
        },
        {
            "style": "Reports",
            "dynamic": "Fact-driven, clean pacing, highlights on costs, deadlines, myths, and key claims.",
            "example": {
                "caption_text": "Der wichtigste Punkt ist die Bewerbungsfrist.",
                "highlight_words": ["wichtigste", "Bewerbungsfrist"],
                "emojis": [{"image": "clock.png", "start": 4.0, "end": 5.2, "matched_word": "Bewerbungsfrist"}],
            },
        },
        {
            "style": "Interview",
            "dynamic": "Natural question-answer rhythm, highlights on decisions, roles, names, and turning points.",
            "example": {
                "caption_text": "Warum hast du dich fuer den Master entschieden?",
                "highlight_words": ["Warum", "Master"],
                "emojis": [{"image": "thinking-removebg-preview.png", "start": 7.2, "end": 8.4, "matched_word": "Warum"}],
            },
        },
        {
            "style": "Day in a Life",
            "dynamic": "Chronological and everyday, calm cuts, highlights on routine, campus, housing, and time.",
            "example": {
                "caption_text": "Morgens geht es erst in die Bib und danach zur Mensa.",
                "highlight_words": ["Morgens", "Bib", "Mensa"],
                "emojis": [{"image": "woman-student.png", "start": 10.5, "end": 11.8, "matched_word": "Bib"}],
            },
        },
    ]


def _normalize_highlights(editing_script):
    for clip in editing_script.get("clips", []):
        highlights = clip.get("highlight_words", [])
        if isinstance(highlights, str):
            highlights = [highlights]
        clip["highlight_words"] = [str(word).strip() for word in highlights if str(word).strip()]
    return editing_script


def _sanitize_emojis(editing_script, emoji_mapping):
    allowed_images = set(emoji_mapping)
    last_emoji_start = -999.0

    for clip in editing_script.get("clips", []):
        valid_emojis = []
        clip_start = float(clip.get("start", 0))
        clip_end = float(clip.get("end", clip_start))

        for emoji in clip.get("emojis", []):
            image_name = emoji.get("image")
            if image_name not in allowed_images:
                continue
            matched_word = str(emoji.get("matched_word", "")).strip()
            if not matched_word:
                continue
            emoji_path = os.path.join(EMOJI_ASSET_DIR, image_name)
            if not os.path.exists(emoji_path):
                continue

            start = float(emoji.get("start", clip_start))
            end = float(emoji.get("end", min(start + 1.2, clip_end)))
            start = max(clip_start, min(start, clip_end))
            end = max(start + 0.2, min(end, clip_end))

            if start - last_emoji_start < 3.0:
                continue

            valid_emojis.append({
                "image": image_name,
                "path": emoji_path,
                "start": start,
                "end": end,
                "matched_word": matched_word,
            })
            last_emoji_start = start

        clip["emojis"] = valid_emojis

    return editing_script


def generate_editing_script(transcript):
    if not LITELLM_MODEL or not LITELLM_API_KEY:
        logging.error("LITELLM_MODEL or LITELLM_API_KEY not found.")
        return None

    style_guide = get_style_guide()
    emoji_mapping = load_emoji_mapping()
    transcript_text = format_transcript_for_llm(transcript)

    system_prompt = f"""You are the brain of a Legmon-style short-form video editor. Your output MUST be a valid JSON object.

Core rules:
1. Analyze the transcript and split it into logical short clips.
2. Do not cut or shorten the video. Use original transcript timestamps only; the renderer keeps the full original video length.
3. Choose one Legmon caption style per clip: legmon_lustig, legmon_reports, legmon_interview, or legmon_day_in_a_life.
4. Add highlight_words for topic changes and core terms. The renderer rotates highlighted words through these colors: {", ".join(HIGHLIGHT_COLORS)}; all other caption words stay white. Do not output color values.
5. Use emojis only from the fixed mapping below. Trigger an emoji only when a spoken word or close synonym clearly matches the mapping. Never trigger emojis more often than every 3-4 seconds across the whole video. Output the exact image filename from emoji_mapping.json.
6. Keep emojis sparse. If in doubt, leave emojis empty for that clip.
7. Keep caption_text punchy. Each rendered caption box is capped at 24 characters including spaces. Prefer 1-3 short words, or at most 3 short words plus 1 long word. Avoid filler so fast speech stays readable and synced.

Available transitions and caption styles:
{json.dumps(style_guide, indent=2)}

Fixed emoji mapping source:
Path: {EMOJI_MAPPING_PATH}
Emoji asset folder: {EMOJI_ASSET_DIR}
Allowed mapping JSON:
{json.dumps(emoji_mapping, indent=2, ensure_ascii=False)}

Legmon few-shot style references:
{json.dumps(_legmon_few_shots(), indent=2, ensure_ascii=False)}

JSON output structure:
- music: Suggest a mood and master volume.
- clips: List of clip objects. Each clip MUST include:
  - start, end: Absolute timestamps in the original video.
  - transition_in: Transition name for this clip.
  - transition_sfx: Sound effect name.
  - caption_style: One of the Legmon caption styles.
  - caption_text: Caption text for this segment.
  - caption_sfx: Sound effect name for caption entrance.
  - highlight_words: Important exact words or short phrases from caption_text.
  - emojis: List of emoji objects. Each emoji object MUST include:
    - image: Exact filename from emoji_mapping.json, e.g. "money-with-wings.png".
    - start, end: Absolute timestamps in the original video. Keep each visible about 0.8-1.4 seconds.
    - matched_word: Spoken word or close synonym that justified this emoji.

Example clip:
{{"start": 5.2, "end": 9.8, "transition_in": "slide_left", "transition_sfx": "whoosh", "caption_style": "legmon_reports", "caption_text": "Diese Frist entscheidet alles.", "caption_sfx": "tick", "highlight_words": ["Frist", "entscheidet"], "emojis": [{{"image": "clock.png", "start": 5.6, "end": 6.8, "matched_word": "Frist"}}]}}
"""

    user_prompt = f"""Create a complete Legmon-style JSON editing script for this transcript.

Transcript Text: "{transcript_text}"

Word-level Timestamps:
{json.dumps(transcript, indent=2, ensure_ascii=False)}

Return JSON only."""

    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"Generating Legmon editing script with LLM (Attempt {attempt + 1}/{max_retries})...")
            completion_kwargs = {
                "model": LITELLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "api_key": LITELLM_API_KEY,
                "temperature": 0.6,
                "response_format": {"type": "json_object"},
            }
            if LITELLM_API_BASE:
                completion_kwargs["api_base"] = LITELLM_API_BASE

            response = completion(**completion_kwargs)
            script_text = response.choices[0].message.content
            if script_text.startswith("```json"):
                script_text = script_text[7:-4]
            script_json = json.loads(script_text)
            script_json = _normalize_highlights(script_json)
            script_json = _sanitize_emojis(script_json, emoji_mapping)
            logging.info("Successfully generated and parsed Legmon editing script.")
            return script_json
        except litellm.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error("Max retries exceeded. Failed to bypass rate limit.")
                raise e
        except Exception as e:
            logging.error(f"Failed to generate or parse LLM script: {e}")
            if "response" in locals():
                logging.error(f"LLM raw response: {response.choices[0].message.content}")
            return None
