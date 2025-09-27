
import os
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import logging

logging.basicConfig(level=logging.INFO)

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

def transcribe_audio(audio_path):
    """Transcribes audio and returns word-level timestamps."""
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found at: {audio_path}")
        return None
    try:
        logging.info("Loading transcription model...")
        # Using a smaller, faster model for the MVP. Can be swapped for "large-v3".
        # Using "cpu" and "int8" for wider compatibility without a powerful GPU.
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        
        logging.info("Starting transcription...")
        segments, _ = model.transcribe(
            audio_path,
            word_timestamps=True
        )
        
        word_level_transcript = []
        for segment in segments:
            for word in segment.words:
                word_level_transcript.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
        
        logging.info("Transcription complete.")
        # Clean up the temporary audio file
        os.remove(audio_path)
        logging.info(f"Removed temporary audio file: {audio_path}")
        
        return word_level_transcript
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        return None


