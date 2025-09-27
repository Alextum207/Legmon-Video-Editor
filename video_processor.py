import os
import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip,
    concatenate_videoclips, vfx, ColorClip, CompositeAudioClip
)
from moviepy.audio.fx.all import audio_normalize
from moviepy.video.compositing.CompositeVideoClip import clips_array
from PIL import Image, ImageDraw, ImageFont
import logging
import textwrap

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def create_caption_image(text, style_name, video_size):
    font_size = int(video_size[1] / 20)
    img = Image.new('RGBA', video_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    max_chars = int(len(text) * (video_size[0] * 0.8) / text_width) if text_width > 0 else 25
    wrapped_text = textwrap.fill(text, width=max_chars if max_chars > 0 else 25)

    if style_name == "modern_lower_third":
        bar_h = int(font_size * 1.5)
        bar_y = video_size[1] * 0.8
        draw.rounded_rectangle([(0, bar_y), (video_size[0], bar_y + bar_h)], radius=10, fill=(0, 0, 0, 178))
        draw.text((video_size[0]/2, bar_y + bar_h/2), wrapped_text, font=font, fill='white', anchor="mm", align="center")
    elif style_name == "pop_up_caption":
        text_bbox = draw.textbbox((0,0), wrapped_text, font=font, align="center")
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        padding = int(font_size * 0.5)
        shape_x0 = (video_size[0] - text_w) / 2 - padding
        shape_y0 = (video_size[1] - text_h) / 2 - padding
        shape_x1 = (video_size[0] + text_w) / 2 + padding
        shape_y1 = (video_size[1] + text_h) / 2 + padding
        draw.rounded_rectangle([(shape_x0, shape_y0), (shape_x1, shape_y1)], radius=padding, fill="#FF5A5F")
        draw.text((video_size[0]/2, video_size[1]/2), wrapped_text, font=font, fill='white', anchor="mm", align="center")
    else: # Minimalist
        draw.text((video_size[0]/2, video_size[1] * 0.5), wrapped_text, font=font, fill='white', anchor="mm", align="center", stroke_width=2, stroke_fill='black')

    temp_img_path = f"temp_caption_{hash(text)}.png"
    img.save(temp_img_path)
    return temp_img_path

def render_video(original_video_path, editing_script, assets, output_path):
    logging.info("Starting final video rendering process...")
    base_clip = VideoFileClip(original_video_path)
    temp_files = []
    rendered_scenes = []

    # Create each scene as a self-contained clip
    for i, clip_info in enumerate(editing_script['clips']):
        start, end = clip_info['start'], clip_info['end']
        duration = end - start
        if duration <= 0: continue

        # Determine the video source for this scene (base clip or B-roll)
        video_source_clip = base_clip
        if clip_info.get('b_roll_keyword') and clip_info['b_roll_keyword'] in assets['b_roll']:
            b_roll_path = assets['b_roll'][clip_info['b_roll_keyword']]
            video_source_clip = VideoFileClip(b_roll_path).resize(height=base_clip.size[1])
            video_source_clip = video_source_clip.crop(x_center=video_source_clip.size[0]/2, width=base_clip.size[0])
        
        scene_video = video_source_clip.subclip(start, end)

        # Create and overlay caption
        caption_img_path = create_caption_image(clip_info["caption_text"], clip_info["caption_style"], base_clip.size)
        temp_files.append(caption_img_path)
        caption_clip = ImageClip(caption_img_path).set_duration(duration)
        
        scene_with_caption = CompositeVideoClip([scene_video, caption_clip], size=base_clip.size)
        rendered_scenes.append(scene_with_caption)

    # Assemble scenes with transitions
    final_clips = []
    transition_duration = 0.5 # seconds
    for i, scene in enumerate(rendered_scenes):
        if i == 0:
            final_clips.append(scene)
            continue
        
        prev_clip_len = final_clips[-1].duration
        # Simple Cross-Fade Transition
        final_clips[-1] = final_clips[-1].fadeout(transition_duration)
        scene = scene.fadein(transition_duration)
        final_clips.append(scene)

    final_video = concatenate_videoclips(final_clips).set_position(('center', 'center'))

    # Composite audio
    # (A more advanced implementation would sync SFX with transitions)
    all_audio = [base_clip.audio.fx(audio_normalize)]
    if assets.get('music'):
        music_clip = AudioFileClip(assets['music']).fx(audio_normalize).volumex(0.1)
        if music_clip.duration > final_video.duration:
            music_clip = music_clip.subclip(0, final_video.duration)
        all_audio.append(music_clip)

    final_video.audio = CompositeAudioClip(all_audio)

    logging.info(f"Writing final video to {output_path}...")
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    for f in temp_files: os.remove(f)
    logging.info("Final video rendering complete!")