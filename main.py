
import os
import argparse
import json
import logging
import PIL.Image

if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = 1

import transcriber
import llm_handler
import asset_manager
import video_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_VIDEO_DIR = os.path.join(BASE_DIR, "roh_videos")
FINISHED_VIDEO_DIR = os.path.join(BASE_DIR, "finished_videos")
TEMP_OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
DEFAULT_OUTPUT_NAME = "final_corrected_video.mp4"


def ensure_project_directories():
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    os.makedirs(FINISHED_VIDEO_DIR, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)


def resolve_input_video(input_video):
    if input_video:
        candidate = input_video
        if not os.path.isabs(candidate):
            candidate = os.path.join(RAW_VIDEO_DIR, candidate)
        return os.path.abspath(candidate)

    raw_videos = [
        os.path.join(RAW_VIDEO_DIR, file_name)
        for file_name in os.listdir(RAW_VIDEO_DIR)
        if os.path.splitext(file_name)[1].lower() in VIDEO_EXTENSIONS
    ]

    if len(raw_videos) == 1:
        return os.path.abspath(raw_videos[0])
    if not raw_videos:
        raise FileNotFoundError(f"No video found in {RAW_VIDEO_DIR}. Add a raw video there or pass --input_video.")

    file_list = ", ".join(os.path.basename(path) for path in raw_videos)
    raise ValueError(f"Multiple videos found in {RAW_VIDEO_DIR}: {file_list}. Pass --input_video with the exact filename.")


def resolve_output_path(output_name):
    output_name = output_name or DEFAULT_OUTPUT_NAME
    if os.path.isabs(output_name):
        return output_name
    return os.path.join(FINISHED_VIDEO_DIR, output_name)


def main():
    ensure_project_directories()
    assets = {'b_roll': {}, 'sfx': {}, 'music': None}
    parser = argparse.ArgumentParser(description="AI Video Editor - Corrective Build")
    parser.add_argument("--input_video", help="Input video path. Relative names are resolved inside roh_videos/. If omitted, the only video in roh_videos/ is used.")
    parser.add_argument("--output_name", default=DEFAULT_OUTPUT_NAME, help="Output filename. Relative names are saved inside finished_videos/.")
    args = parser.parse_args()
    try:
        input_video_path = resolve_input_video(args.input_video)
        output_file_path = resolve_output_path(args.output_name)
    except (FileNotFoundError, ValueError) as e:
        logging.error(e)
        return

    if not os.path.exists(input_video_path):
        logging.error(f"Input video not found: {input_video_path}")
        return

    # --- 1. Transcription ---
    logging.info("--- Step 1: Transcription ---")
    transcript_path = transcriber.find_sidecar_transcript(input_video_path)
    if transcript_path:
        word_transcript = transcriber.transcribe_from_text_file(transcript_path, input_video_path)
    else:
        temp_audio_path = transcriber.extract_audio(input_video_path)
        if not temp_audio_path: return
        word_transcript = transcriber.transcribe_audio(temp_audio_path)
    if not word_transcript: return

    # --- 2. LLM Editing Script Generation ---
    logging.info("--- Step 2: Generating Corrected Editing Script ---")
    editing_script = llm_handler.generate_editing_script(word_transcript)
    if not editing_script: return

    if isinstance(editing_script, str):
        try:
            editing_script = json.loads(editing_script)
        except json.JSONDecodeError:
            logging.error("LLM returned a string that is not valid JSON.")
            return

    # --- 3. Asset Fetching --- #
    logging.info("--- Step 3: Fetching All Required Assets ---")
    # Get B-Roll
    b_roll_keywords = set(c.get('b_roll_keyword') for c in editing_script.get('clips', []) if c.get('b_roll_keyword'))
    for keyword in b_roll_keywords:
        b_roll_path = asset_manager.get_b_roll(keyword)
        if b_roll_path: assets['b_roll'][keyword] = b_roll_path

    # Get SFX
    sfx_to_fetch = set()
    for c in editing_script.get('clips', []):
        if c.get('transition_sfx'): sfx_to_fetch.add(c['transition_sfx'])
        if c.get('caption_sfx'): sfx_to_fetch.add(c['caption_sfx'])
    for sfx_name in sfx_to_fetch:
        sfx_path = asset_manager.get_sfx(sfx_name)
        if sfx_path: assets['sfx'][sfx_name] = sfx_path

    # Get Music
    music_mood = editing_script.get('music', {}).get('mood')
    if music_mood:
        assets['music'] = asset_manager.get_music(music_mood)

    # --- 4. Video Rendering --- #
    logging.info("--- Step 4: Rendering Corrected Video ---")
    video_processor.render_video(
        original_video_path=input_video_path,
        editing_script=editing_script,
        assets=assets,
        output_path=output_file_path,
        transcript=word_transcript,
    )

    logging.info(f"Process complete! Corrected video is at: {output_file_path}")

if __name__ == "__main__":
    main()
