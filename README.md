
# 🎬 Auto-Cut AI: The AI-Powered Video Editor

**Transform raw talking-head videos into dynamic, share-ready social media clips with a single command.**

This tool leverages the power of Large Language Models (LLMs) to automate the entire video editing pipeline, from transcription and content analysis to final rendering with professional effects, captions, and sound design.

![Project Demo GIF](https://your-image-host.com/demo.gif)  
*(Note: You can create and add a demo GIF here to showcase the tool in action!)*

---

## ✨ Features

- **AI-Powered Editing Decisions:** Uses an LLM (via LiteLLM) to analyze the video's transcript and generate a dynamic editing script.
- **Automatic Transcription:** Employs `faster-whisper` for fast and accurate speech-to-text with word-level timestamps.
- **Dynamic Captions & Subtitles:** Generates multiple styles of animated captions, from professional lower-thirds to trendy pop-up text.
- **Intelligent B-Roll:** Automatically finds and inserts relevant stock footage (from Pexels) based on the spoken content.
- **Automated Sound Design:** Downloads and syncs background music and sound effects (from Pixabay) to transitions and animations.
- **Professional Transitions:** Implements cross-fades and other visual effects to create a polished final product.
- **Fully Configurable:** All API keys and model configurations are managed via a simple `.env` file.

---

## ⚙️ How It Works

The tool follows a 4-step, end-to-end pipeline:

1.  **Transcribe & Analyze:** The audio is extracted from the source video and transcribed to generate word-level timestamps. The full transcript is prepared for the AI.

2.  **Generate Editing Script:** The transcript and a style guide are sent to an LLM. The AI analyzes the content and pacing, then returns a structured JSON script outlining every scene, transition, caption style, and B-roll shot.

3.  **Fetch Assets:** The script automatically queries stock media APIs (Pexels, Pixabay) to download the exact B-roll clips, background music, and sound effects specified in the AI's editing script.

4.  **Render Final Video:** The `moviepy` and `Pillow` libraries are used to execute the editing script. The video is composited with animated captions, transitions, B-roll, and audio, then rendered into a final MP4 file.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- `git`

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Auto-Cut-AI.git
cd Auto-Cut-AI
```

### 2. Set Up the Environment

Create a virtual environment and install the required packages.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

You need to provide API keys for the AI model and stock asset providers.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Open the `.env` file and add your keys:
    - `LITELLM_API_KEY`: Your key for the LLM provider (e.g., Mistral, OpenAI).
    - `PEXELS_API_KEY`: Get a free key from [Pexels](https://www.pexels.com/api/).
    - `PIXABAY_API_KEY`: Get a free key from [Pixabay](https://pixabay.com/api/docs/).

---

## 💻 Usage

Place your raw video file (e.g., `my_video.mp4`) in the root directory of the project.

Run the main script with the path to your input video:

```bash
python main.py --input_video your_video_name.mp4
```

You can also specify a custom output name:

```bash
python main.py --input_video your_video.mp4 --output_name awesome_clip.mp4
```

The final, edited video will be saved in the `/output` directory.

---

## 🗺️ Roadmap & Future Work

This project has a lot of potential for growth. Future enhancements could include:

- [ ] **More Advanced Transitions:** Implementing dynamic distortions (`Zoom Blur`, `Whip Pan`) and stylized effects (`Glitch`, `Light Leaks`).
- [ ] **Complex Caption Animations:** Adding `Typewriter` reveals and word-level highlighting.
- [ ] **Web Interface:** Building a simple web UI to upload videos and select styles instead of using the CLI.
- [ ] **Style Customization:** Allowing users to define their own caption styles and transition preferences in a config file.
- [ ] **GPU Acceleration:** Offloading transcription and rendering to a GPU for significant speed improvements.

---
