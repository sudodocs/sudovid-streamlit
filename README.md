# 🖊️ ArchiText
### AI-Powered Script, Voice & Video Engine

ArchiText is a Streamlit application that takes your raw idea or rough notes and turns them into a fully produced YouTube-ready video — complete with a researched script, professional voiceover, and an assembled MP4 built from TMDB stills, Pexels B-roll, Wikipedia images, and auto-generated graphic cards.

It supports three content modes out of the box:
- 🎬 **Film & Series Analysis** — movie/series reviews and commentary
- 🔍 **Tech News & Investigative** — tech incidents, product launches, investigative pieces
- 📚 **Educational Technology** — tutorials, explainers, developer content

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)
- [API Keys You Need](#api-keys-you-need)
- [How to Use — Step by Step](#how-to-use--step-by-step)
  - [Step 1 — Parameters](#step-1--parameters)
  - [Step 2 — Ground Research](#step-2--ground-research)
  - [Step 3 — Generated Script](#step-3--generated-script)
  - [Step 4 — Voiceover](#step-4--voiceover)
  - [Step 5 — Content Bundle](#step-5--content-bundle)
  - [Step 6 — Video Assembly](#step-6--video-assembly)
- [Content Mode Reference](#content-mode-reference)
- [Video Assembly — How Visuals Are Chosen](#video-assembly--how-visuals-are-chosen)
- [File Structure](#file-structure)
- [Known Limitations](#known-limitations)

---

## Requirements

- Python 3.11+
- `ffmpeg` installed on the system (handled automatically on Streamlit Cloud via `packages.txt`)

---

## Installation

```bash
# Clone your repo and navigate into it
git clone https://github.com/your-username/architext.git
cd architext

# Install Python dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

For local runs, ensure `ffmpeg` is installed on your machine:
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH

---

## Deploying to Streamlit Cloud

1. Push your repo to GitHub (must contain `app.py`, `requirements.txt`, and `packages.txt`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `app.py`
4. Deploy — Streamlit Cloud will handle `ffmpeg` via `packages.txt` automatically

Your repo should contain these four files at the root:

```
app.py
requirements.txt
packages.txt
README.md
```

`packages.txt` content (one line):
```
ffmpeg
```

---

## API Keys You Need

| Key | Where to Get | Used In |
|---|---|---|
| **Gemini API Key** | [aistudio.google.com](https://aistudio.google.com) | Research, Script, Bundle, Shot List |
| **TMDB API Key** | [themoviedb.org → Settings → API](https://www.themoviedb.org/settings/api) | Film & Series mode only — movie stills and cast photos |
| **Pexels API Key** | [pexels.com/api](https://www.pexels.com/api) | All modes — cinematic B-roll images |

Wikipedia images (used in Tech and Educational modes) are fetched via a free public API — no key needed.

All keys are entered inside the app. The Gemini key goes in the sidebar. The TMDB and Pexels keys are entered in Tab 6 (Video Assembly) when you reach that step.

---

## How to Use — Step by Step

The app follows a linear six-step workflow. Complete each step in order before moving to the next tab.

---

### Step 1 — Parameters

**Tab: `1. Parameters`**

This is where you define everything about your video before any AI work happens.

1. **Topic or Title** — Enter the name of the film, series, tech event, or educational topic.
   - Examples: `Oppenheimer`, `CrowdStrike Outage July 2024`, `How Transformers Work`

2. **Content Mode** — Select the type of content you are creating:
   - `Film & Series Analysis` — for movie/series reviews
   - `Tech News & Investigative` — for tech breakdowns and incident reports
   - `Educational Technology` — for tutorials and explainer videos

3. **Target Video Length** — Controls how long and detailed the generated script will be:
   - `YouTube Short (< 1 minute)` — punchy, under 150 words
   - `Mid-length (3-8 mins)` — balanced narrative
   - `Deep Dive (10+ mins)` — comprehensive, multiple perspectives

4. **Source Material** *(Film mode only)* — Tell the AI what the film is based on:
   - `Original`, `Book`, `Comic`, `True Event`, or `Remake`

5. **Tuning Matrix** — Sliders that shape the tone and style of the script:
   - *Film mode:* Film Theory Focus, Visual Signature, Adaptation Fidelity, Narrative Tone
   - *Tech mode:* Criticality (Bug → Crisis), User Impact (Niche → Global)
   - *Educational mode:* Knowledge Level (Junior → Architect), Pedagogical Style (Theory → Practical)

6. **Your Unique Angle** — The most important input. Type your own perspective, opinion, or rough draft notes. The AI uses this as the foundation — it will research to support YOUR angle, not generate a generic summary. You can also upload a `.txt` file instead.

7. Click **💾 Save Parameters & Proceed**.

---

### Step 2 — Ground Research

**Tab: `2. Ground Research`**

ArchiText sends your topic and angle to Gemini with web search enabled. It actively looks for facts, data, dates, and context that specifically support the angle you defined — not a generic Wikipedia summary.

1. Click **🔍 Execute Targeted Background Research**
2. Wait 15–30 seconds while Gemini searches the web
3. Expand **View Factual Briefing** to review what was found (with source URLs)
4. Proceed to Tab 3

> **Tip:** If the research misses something important, note it in your script edit in Step 3 — you can always add facts manually.

---

### Step 3 — Generated Script

**Tab: `3. Generated Script`**

Gemini synthesises the research and your angle into a full conversational script structured as:
- **Hook** — punchy opening line
- **Intro** — context and setup
- **Act 1, Act 2, Act 3** — the core of your review/analysis
- **Outro** — conclusion and call to action

**After generation:**
- The script appears in an editable text area
- Edit freely — change wording, add pauses with commas or `---`, cut anything you don't like
- Your edits are used in all subsequent steps
- Click **📥 Download Text Script** to save a `.txt` copy

Click **🚀 Architect Refined Script** to generate, then proceed to Tab 4.

---

### Step 4 — Voiceover

**Tab: `4. Voiceover`**

Converts your script to audio using Microsoft Neural voices via Edge TTS (no paid API needed for this step).

1. **Select a voice** from 12 US English options (6 male, 6 female)
2. **Choose text source:**
   - `Use Generated Script` — uses what you edited in Tab 3
   - `Upload Custom Text File` — upload a `.txt` file if you want to use your own script
3. Review the text in the preview area — make any final edits
4. Click **🔊 Generate Voiceover** (takes 10–30 seconds)
5. Listen to the preview audio player
6. Download the `.mp3` if needed

> The audio file is automatically saved for use in Tab 6. You must complete this step before video assembly.

---

### Step 5 — Content Bundle

**Tab: `5. Content Bundle`**

Generates everything you need to publish on YouTube:

- **Viral Title** — high-CTR YouTube title
- **Description** — full video description with hook and summary
- **Tags** — 15 SEO-optimised tags (comma separated, ready to paste)
- **Hashtags** — 3–5 relevant hashtags
- **AI Thumbnail Prompt** — a detailed image generation prompt you can paste into Midjourney, DALL-E, or Canva to create a thumbnail

1. Choose whether to base the bundle on the Tab 3 script or the Tab 4 audio text
2. Click **📦 Generate Content Bundle**
3. Copy each field directly into YouTube Studio when uploading

---

### Step 6 — Video Assembly

**Tab: `6. Video Assembly`**

Automatically assembles a complete MP4 video by combining your voiceover audio with visuals sourced and generated based on your content mode.

#### Before you start

Enter your image API keys in the expander at the top of this tab:
- **TMDB API Key** — required if you are in Film & Series mode
- **Pexels API Key** — required for all modes

#### Step-by-step

1. Click **🎬 Generate Shot List**
   - Gemini reads your final script and plans a timed list of 8–20 clips
   - Each clip has a type, duration, and source instruction
   - Expand **View Shot List** to review what was planned

2. Click **🚀 Start Video Assembly**
   - Assembly runs in the background — **keep the browser tab open**
   - A live progress bar updates every 3 seconds showing the current clip being built
   - Total time is typically 3–6 minutes depending on video length

3. When complete, click **📥 Download MP4** to save your video

4. Use **🔄 Reset & Assemble Again** if you want to regenerate with a different shot list

> **Important:** Do not close or refresh the browser tab while assembly is running. Progress is tracked via a background thread — closing the tab will not stop the process, but you will lose the progress display and the download button.

---

## Content Mode Reference

### Film & Series Analysis
Uses TMDB to pull official posters, cinematic scene backdrops, and cast headshots. Character names are pulled directly from the generated script package so cast photo lookups are accurate. Text cards show ratings, box office figures, and review highlights.

### Tech News & Investigative
Does not use TMDB. Sources visuals from Pexels (server rooms, code screens, offices) and Wikipedia (company logos, product images, person photos). Stat cards emphasise impact numbers like users affected, downtime hours, and financial damage.

### Educational Technology
Does not use TMDB. Sources concept B-roll from Pexels. Generates code cards (dark VS Code-style panels with syntax colour-coding) when the script references actual code. Text cards highlight definitions, key terms, and takeaways. Chapter cards mark section transitions using the script outline.

---

## Video Assembly — How Visuals Are Chosen

Every clip in the assembled video is one of these types:

| Type | Description | Modes |
|---|---|---|
| `tmdb_poster` | Official film poster | Film only |
| `tmdb_backdrop` | Cinematic scene still from TMDB | Film only |
| `tmdb_cast` | Actor headshot from TMDB | Film only |
| `pexels_broll` | Cinematic B-roll from Pexels | All modes |
| `wiki_image` | Wikipedia article thumbnail | Tech, Educational |
| `text_card` | Styled graphic card with text | All modes |
| `stat_card` | Large-number impact card | All modes |
| `code_card` | Dark-themed code snippet panel | Educational only |
| `chapter_card` | Act/section transition card | All modes |

If any image fetch fails (network error, image not found), ArchiText automatically falls back — first trying a Pexels search on the topic, then generating a text card. The video always completes regardless of individual fetch failures.

All clips use a gentle Ken Burns zoom effect (alternating zoom-in and zoom-out) so the video never feels like a static slideshow.

---

## File Structure

```
architext/
├── app.py              # Main application — all logic in one file
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Cloud (ffmpeg)
└── README.md           # This file
```

---

## Known Limitations

- **Browser tab must stay open during video assembly.** The background thread continues running if the tab is closed, but the progress display and download button will not reappear.
- **Streamlit Cloud free tier** has limited CPU, so video rendering (the final ffmpeg encode) may be slow on longer videos. A Deep Dive (10+ min) video with many clips may take 8–12 minutes to render.
- **TMDB stills** are official press images. They may not include every scene or character. Gaps are filled automatically with Pexels B-roll.
- **Edge TTS** requires an internet connection. It does not work offline.
- **Session state is lost on browser refresh.** Use the **Reset All Steps** button in the sidebar for a clean restart rather than refreshing the page.
- **YouTube Shorts mode** generates a very short script (under 150 words). The video assembly step still works, but the resulting video will be brief — adjust clip durations manually in the shot list if needed.
