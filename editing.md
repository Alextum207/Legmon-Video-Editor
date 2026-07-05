# Legmon Video Editing Guide

This file is the shared editing memory for all video editing agents working on this project. Follow it before changing caption timing, visual effects, sound effects, color, or the render pipeline.

## Core Goal

Create short-form interview/reel edits that feel like modern Shorts/Reels:

- fast and present, but not chaotic
- strong captions with accurate sync
- intentional sound effects at meaningful moments
- no random transitions
- no distracting zoom behavior
- preserve the original repo's good coloring/caption look unless explicitly asked to change it

## Always Preserve

- Keep the optimized caption synchronization logic.
- Keep word-level timing whenever available.
- Prefer Speechmatics transcription when `SPEECHMATICS_API_KEY` is configured.
- Fall back to faster-whisper only when Speechmatics is unavailable or fails.
- Preserve the existing caption style/coloring from the codebase unless the user explicitly asks for a visual redesign.
- Export final videos to `finished_videos`.
- Keep everything local-first on Windows.

## Captions

### Do

- Use accurate word-level timestamps.
- Keep captions short and readable.
- Keep the existing Legmon caption style as the baseline.
- Highlight exactly every second caption when a meaningful word can be selected.
- Highlight one meaningful word or compact term, using the repo scoring rules.
- Keep question marks and punctuation attached to words, because later editing logic uses them.

### Don't

- Do not replace the working caption algorithm with a simpler generic subtitle renderer.
- Do not make captions drift from the spoken audio.
- Do not make huge text blocks.
- Do not highlight every word.
- Do not change coloring just because a new effect is added.

### Coloring Rule

- If no category-based term is found, use the strongest non-stopword fallback so the rhythm stays consistent.
- Rotate strictly through the four configured colors in order: red, yellow, mint, green.
- A color must not repeat until all other configured colors have been used once.
- Do not choose colors based on event type, speaker, emotion, or timing.

## Sound Effects

Sound effects should be present and noticeable like Shorts/Reels, but still motivated by the content.

### Available Local SFX

- `opening_hit`: `dragon_studio_ding.mp3`
- `topic_wosh`: `deep_wosh.mp4`
- `topic_riser`: `risers.mp4`

### General Rules

Do use SFX:

- at the beginning of an edit
- on clear topic changes
- at the end of a thought/topic when the edit needs a release
- exactly 0.10 seconds before the relevant highlighted word when possible

Do not use SFX:

- on questions
- during serious technical explanations
- during dense factual explanations where the sound would feel unserious
- randomly every few seconds without a content reason
- under every caption
- so quietly that only the ping is audible

Timing rule:

- The ding/bing sound is only the opening hook unless the user explicitly requests keyword dings again.
- Keep at least about 1.25 seconds between SFX unless the user explicitly asks for denser hype pacing.
- Wosh/riser can be longer, but should still be tied to a topic change, buildup, or ending.

### Ping / Ding

Use ping:

- only at the very beginning of the video

Avoid ping:

- after the opening hook
- on questions
- on long explanatory sentences
- when another bigger SFX is already carrying the moment

### Deep Wosh

Use deep wosh:

- on topic changes
- when a thought is closing
- when the rhythm moves into a new section
- as an exit/outro feeling from one topic

Avoid deep wosh:

- on tiny words
- inside a serious technical explanation
- when it makes the speaker feel interrupted

### Riser

Use riser:

- before a strong ending
- into a reveal or conclusion
- before a punchy final sentence
- when the speaker builds toward an important point

Avoid riser:

- under casual filler
- under questions
- too often in a short clip

## Zooms

Current rule: zooms are answerer punch-ins, not random center zooms.

- use snap zooms, not slow drifting zooms
- when an interviewer asks a question, punch in hard as the answer starts
- identify which side/person belongs to the answerer, then zoom to that person only
- hold the punch-in only while that answerer is actually speaking
- end the punch-in as soon as the interviewer starts speaking again, even if the interviewer talks about the answerer
- when speaker labels are available, use the active speaker label to end the punch-in, not mentions of a person in the text
- in two-person interviews, do not crop the middle between both people
- frame the answerer like a Shorts close-up: tight side crop, speaker large, captions still centered as an overlay
- jump from 100% to about 155% and hold for the full answer
- keep captions rendered on top of the final zoomed/cropped picture
- never add random zooms
- never use slow automatic zoom-ins across long sentences
- do not zoom on questions or serious technical explanations
- do not add small 105% keyword zooms unless explicitly requested again

## Speaker Tracking

Use speaker/face tracking only when it improves framing.

- Prefer local OpenCV/MediaPipe-style face detection.
- For already vertical source footage, skip tracking unless there is an obvious framing problem.
- For wide interview footage, sample faces roughly every 0.25 seconds.
- If speaker labels exist, map the active speaker to the nearest learned face position.
- If speaker labels do not exist, follow the largest detected face as a fallback.
- Use snap thresholds for hard speaker changes and smoothing/deadzone for normal movement.
- Do not let tracking fight the caption readability or create random drifting camera motion.

## Interviews

For interviews:

- SFX should mainly support answers, not questions.
- Do not place sounds on the interviewer asking a question.
- The ding/bing sound is only the opening hook unless explicitly requested again.
- Use non-bing sounds sparingly when the interviewee gives a strong answer, fact, compact insight, or list.
- Avoid comedic or hype sounds when the answer is serious, academic, or technical.
- Keep the edit respectful when the topic is professional or thoughtful.
- Zooms belong to the current answerer, not to the person being mentioned by the interviewer.

## Emoji Overlays

- Show `thinking-removebg-preview.png` when words like "denken", "nachdenken", "ueberlegen", or "gruebeln" are spoken.
- Treat thinking/nachdenken as priority emoji moments so they can appear even when another emoji recently appeared.
- Do not use emoji overlays on questions unless the user explicitly asks for that style.

## Serious Technical Explanations

When the speaker is explaining something complex or fachlich:

- reduce SFX density
- prioritize caption clarity
- avoid hype effects
- avoid zooms
- let the speaker's point breathe

## Visual Style

### Do

- Preserve the original video color and repo coloring unless asked otherwise.
- Keep the person natural and readable.
- Keep edits clean and focused on the speaker.
- Use effects only when they strengthen the spoken point.

### Don't

- Do not degrade color grading.
- Do not add random transitions.
- Do not add glitch, whip pan, light leaks, or heavy stylized effects unless explicitly requested.
- Do not make the edit feel like a generic template.

## Pipeline Preference

1. If a matching subtitle/transcript file is provided, use its words as the perfected/correct text reference.
2. For TXT/SRT/VTT subtitle files, use the file as text reference only, then create fresh internal word synchronization from the video audio via Speechmatics when configured.
3. Otherwise use Speechmatics for transcription.
4. If Speechmatics fails, fall back to faster-whisper.
5. Generate the editing script.
6. Use local SFX assets.
7. Render to `finished_videos`.
8. Verify that captions are synced and all intended SFX are audible.

## TurboScribe Workflow

- Upload the rough-cut video to `https://turboscribe.ai/de/dashboard`.
- Export SRT/VTT with compact segments; around 5 to 8 words per segment is a good starting point.
- Drop the rough-cut video and TurboScribe SRT/VTT into the local UI.
- Treat the SRT/VTT words as the spelling/text reference, not as final word timing.
- Re-sync words internally from the video audio before caption rendering.
- Save the final rendered video in `finished_videos`.

## Agent Checklist Before Rendering

- Are captions synced to speech?
- Are question captions marked with `?` so SFX can avoid them?
- Are sound effects placed on meaningful moments, not randomly?
- Is the opening hit present?
- Are deep wosh and riser audible enough compared to the original voice?
- Are answer punch-in zooms applied only when they are part of the requested workflow?
- Is the original caption/color style preserved?
- Is the output going to `finished_videos`?

## Agent Checklist After Rendering

- Confirm the output path.
- Check the render completed successfully.
- Listen or inspect enough of the audio mix to confirm non-ping SFX are actually audible.
- Do not claim effects were added unless they are present in the rendered output.
- If the result is experimental, say so clearly.
