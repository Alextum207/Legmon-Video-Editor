
import os
import argparse
import logging
import transcriber
import llm_handler
import asset_manager
import video_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    assets = {'b_roll': {}, 'sfx': {}, 'music': None}
    parser = argparse.ArgumentParser(description="AI Video Editor - Corrective Build")
    parser.add_argument("--input_video", required=True, help="Path to the input video file.")
    parser.add_argument("--output_name", default="final_corrected_video.mp4", help="Name of the output video file.")
    args = parser.parse_args()

    # --- 1. Transcription ---
    logging.info("--- Step 1: Transcription ---")
    temp_audio_path = transcriber.extract_audio(args.input_video)
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
    output_file_path = os.path.join("output", args.output_name)
    video_processor.render_video(
        original_video_path=args.input_video,
        editing_script=editing_script,
        assets=assets,
        output_path=output_file_path
    )

    logging.info(f"Process complete! Corrected video is at: {output_file_path}")

if __name__ == "__main__":
    main()
