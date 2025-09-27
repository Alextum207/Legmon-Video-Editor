
import os
import json
from dotenv import load_dotenv
from litellm import completion
import logging
import litellm

load_dotenv()

LITELLM_MODEL = os.getenv("LITELLM_MODEL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

def get_style_guide():
    """Returns a dictionary of all available styles and transitions."""
    return {
        "transitions": [
            {"name": "slide_left", "sfx": "whoosh"},
            {"name": "slide_right", "sfx": "whoosh"},
            {"name": "push_up", "sfx": "whoosh"},
            {"name": "push_down", "sfx": "whoosh"},
            {"name": "flash_frame", "sfx": "light_sfx"}
        ],
        "caption_styles": [
            {"name": "modern_lower_third", "font": "Montserrat Bold", "animation_in": "slide_up", "sfx_in": "whoosh"},
            {"name": "kinetic_type", "font": "Open Sans Semi-Bold", "animation_in": "fade_in"},
            {"name": "pop_up_caption", "font": "Roboto Medium", "animation_in": "pop_up", "sfx_in": "pop"},
            {"name": "minimalist_overlay", "font": "Helvetica Neue", "animation_in": "fade_in"}
        ]
    }

def format_transcript_for_llm(transcript):
    """Formats the word-level transcript into a single string for the LLM prompt."""
    return "".join([word_info["word"] for word_info in transcript])

def generate_editing_script(transcript):
    if not LITELLM_MODEL or not LITELLM_API_KEY:
        logging.error("LITELLM_MODEL or LITELLM_API_KEY not found.")
        return None

    style_guide = get_style_guide()
    transcript_text = format_transcript_for_llm(transcript)

    system_prompt = f"""You are an expert video editor AI creating a dynamic, professional UGC-style video. Your output MUST be a valid JSON object.

1.  **Analyze the Transcript:** Identify logical segments (sentences or short phrases).
2.  **Do NOT Cut the Video:** The original video is one continuous take. You will not make any cuts. Instead, you will define clips that span the entire video and apply transitions BETWEEN them.
3.  **Assign Transitions:** For each segment after the first, choose a dynamic transition from the list. The transition marks the visual change from one style of caption/effect to the next.
4.  **Assign Caption Styles:** For each segment, choose a caption style from the list. Vary the styles to keep it interesting.
5.  **Add Sound Effects:** Based on your choices, specify which sound effects (sfx) to use for transitions and caption animations.

**AVAILABLE STYLES & TRANSITIONS:**
{json.dumps(style_guide, indent=2)}

**JSON OUTPUT STRUCTURE:**
- `music`: Suggest a mood (e.g., 'upbeat background') and a master volume.
- `clips`: A list of sequential clip objects. Each object MUST have:
  - `start`, `end`: Timestamps from the transcript.
  - `transition_in`: Name of the transition to use at the START of this clip (except for the first clip).
  - `transition_sfx`: Name of the sound effect for the transition.
  - `caption_style`: Name of the caption style to use for this segment.
  - `caption_text`: The transcribed text for this segment.
  - `caption_sfx`: Name of the sound effect for the caption's entrance.

Example clip object:
{{ "start": 5.2, "end": 9.8, "transition_in": "slide_left", "transition_sfx": "whoosh", "caption_style": "pop_up_caption", "caption_text": "This is so cool!", "caption_sfx": "pop" }}
"""

    user_prompt = f"""Based on the rules and styles provided, create a complete JSON editing script for the following transcript data.

Transcript Text: "{transcript_text}"

Word-level Timestamps (for your reference):
{json.dumps(transcript, indent=2)}

Generate the JSON script now."""

    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"Generating new professional editing script with LLM (Attempt {attempt + 1}/{max_retries})...")
            response = completion(
                model=LITELLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_key=LITELLM_API_KEY,
                temperature=0.6,
                response_format={"type": "json_object"}
            )
            script_text = response.choices[0].message.content
            if script_text.startswith("```json"): 
                script_text = script_text[7:-4]
            script_json = json.loads(script_text)
            logging.info("Successfully generated and parsed professional editing script.")
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
            if 'response' in locals():
                logging.error(f"LLM raw response: {response.choices[0].message.content}")
            return None
