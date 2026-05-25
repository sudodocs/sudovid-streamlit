# 🎬 SudoVid
### AI-Powered Script, Voice & Video Engine

SudoVid is a Streamlit application that takes your raw idea or rough notes and turns them into a fully produced YouTube-ready video — complete with a researched script, professional voiceover, and an assembled MP4 built from official trailer clips, TMDB stills, Pexels B-roll, Wikipedia images, and auto-generated graphic cards.

It supports three content modes and two output formats:
- 🎬 **Film & Series Analysis** — movie/series reviews and commentary
- 🔍 **Tech News & Investigative** — tech incidents, product launches, investigative pieces
- 📚 **Educational Technology** — tutorials, explainers, developer content
- 📱 **YouTube Shorts (9:16)** or 🖥️ **Long-form (16:9)** — selected automatically from your video length choice

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
- [Video Assembly — Visual Types by Mode](#video-assembly--visual-types-by-mode)
- [Shorts vs Long-Form](#shorts-vs-long-form)
- [Credit Overlay](#credit-overlay)
- [File Structure](#file-structure)
- [Known Limitations](#known-limitations)

---

## Requirements

- Python 3.11+
- `ffmpeg` installed on the system (handled automatically on Streamlit Cloud via `packages.txt`)

---

## Installation

```bash
git clone https://github.com/your-username/sudovid.git
cd sudovid

pip install -r requirements.txt

streamlit run app.py
```

For local runs, install `ffmpeg` on your machine:
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH

---

## Deploying to Streamlit Cloud

1. Push your repo to GitHub (must contain `app.py`, `requirements.txt`, `packages.txt`, and `README.md`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `app.py`
4. Deploy — Streamlit Cloud installs `ffmpeg` automatically via `packages.txt`

Your repo root should contain these files:

```
app.py
requirements.txt
packages.txt
README.md
```

`packages.txt` must contain one line:
```
ffmpeg
```

---

## API Keys You Need

| Key | Where to Get | Used In |
|---|---|---|
| **Gemini API Key** | [aistudio.google.com](https://aistudio.google.com) | Research, Script, Bundle, Shot List |
| **TMDB API Key** | [themoviedb.org → Settings → API](https://www.themoviedb.org/settings/api) | Film & Series mode — movie stills and cast photos |
| **Pexels API Key** | [pexels.com/api](https://www.pexels.com/api) | All modes — cinematic B-roll images |

Wikipedia images (Tech and Educational modes) and YouTube trailer search (Film mode via yt-dlp) require no additional keys.

The Gemini key goes in the sidebar. TMDB and Pexels keys are entered in Tab 6 when you reach the video assembly step.

---

## How to Use — Step by Step

The app follows a linear six-step workflow. Complete each step in order before moving to the next tab.

---

### Step 1 — Parameters

**Tab: `1. Parameters`**

Define everything about your video before any AI work begins.

1. **Topic or Title** — the film, series, tech event, or educational concept.
   - Examples: `Project Hail Mary`, `CrowdStrike Outage July 2024`, `How Transformers Work`

2. **Content Mode** — selects the research persona and visual strategy:
   - `Film & Series Analysis`
   - `Tech News & Investigative`
   - `Educational Technology`

3. **Target Video Length** — controls script length AND output video format:
   - `YouTube Short (< 1 minute)` → 9:16 vertical canvas (1080×1920), 30–58s total
   - `Mid-length (3-8 mins)` → 16:9 landscape canvas (1280×720)
   - `Deep Dive (10+ mins)` → 16:9 landscape canvas (1280×720)

4. **Source Material** *(Film mode only)* — `Original`, `Book`, `Comic`, `True Event`, or `Remake`

5. **Tuning Matrix** — sliders shaping the script's tone and style:
   - *Film:* Film Theory Focus, Visual Signature, Adaptation Fidelity, Narrative Tone
   - *Tech:* Criticality (Bug → Crisis), User Impact (Niche → Global)
   - *Educational:* Knowledge Level (Junior → Architect), Pedagogical Style (Theory → Practical)

6. **Your Unique Angle** — your own perspective, opinion, or rough notes. The AI researches to support this angle specifically, not to produce a generic summary. Type directly or upload a `.txt` file.

7. Click **💾 Save Parameters & Proceed**

---

### Step 2 — Ground Research

**Tab: `2. Ground Research`**

SudoVid sends your topic and angle to Gemini with Google Search enabled. It looks for facts, dates, quotes, and data that strengthen your specific angle — not a generic Wikipedia summary.

1. Click **🔍 Execute Targeted Background Research**
2. Wait 15–30 seconds
3. Expand **View Factual Briefing** to review findings (with source URLs)
4. Proceed to Tab 3

> **Tip:** If the research misses something specific, add it manually when editing the script in Step 3.

---

### Step 3 — Generated Script

**Tab: `3. Generated Script`**

Gemini synthesises the research and your angle into a full conversational script:

- **Hook** — punchy opening line
- **Intro** — context and setup
- **Acts 1, 2, 3** — the core of your analysis
- **Outro** — conclusion and call to action

The script appears in an editable text area. Edit freely — change wording, add pauses with commas or `---`, cut sections you don't want. Everything you type here flows into subsequent steps.

Click **📥 Download Text Script** to save a `.txt` copy at any point.

---

### Step 4 — Voiceover

**Tab: `4. Voiceover`**

Converts your script to audio using Microsoft Neural voices via Edge TTS (no paid API required).

1. **Select a voice** — 12 US English options (6 male, 6 female)
2. **Choose source** — use the generated script from Tab 3, or upload a custom `.txt`
3. Review and tweak the text in the preview area
4. Click **🔊 Generate Voiceover** (10–30 seconds)
5. Preview the audio player and download the `.mp3` if needed

The audio file is saved automatically and passed to Tab 6. You must complete this step before video assembly.

---

### Step 5 — Content Bundle

**Tab: `5. Content Bundle`**

Generates your full YouTube publishing package:

- **Viral Title** — high-CTR YouTube title
- **Description** — full video description with hook and summary
- **Tags** — 15 SEO-optimised tags ready to paste
- **Hashtags** — 3–5 relevant hashtags
- **AI Thumbnail Prompt** — detailed image generation prompt for Midjourney, DALL-E, or Canva

Choose whether to base the bundle on the Tab 3 script or the Tab 4 audio text, then click **📦 Generate Content Bundle**.

---

### Step 6 — Video Assembly

**Tab: `6. Video Assembly`**

Assembles the final MP4 by combining your voiceover with visuals sourced and generated for your content mode and format.

#### Before you start

Enter your keys in the API Keys expander:
- **TMDB API Key** — required for Film & Series mode
- **Pexels API Key** — required for all modes

Set your **Credit Text** — defaults to `"Courtesy: {your topic}"`. This is overlaid on every frame (top centre for Shorts, bottom left for long-form).

#### Workflow

**1. Pre-fetch Trailer** *(Film mode only — optional but strongly recommended)*

Click **🎞️ Pre-fetch Trailer** to search YouTube for the official trailer using `yt-dlp` and download it at 720p. SudoVid extracts 4–8 evenly-spaced segments automatically, skipping the studio logo at the start and the release date slate at the end. The download is cached so re-runs are instant.

If the trailer download fails or is skipped, the shot list falls back to TMDB stills and Pexels B-roll automatically.

**2. Generate Shot List**

Click **🎬 Generate Shot List**. Gemini reads your full script and plans a timed list of clips — each assigned a visual type, duration, and source instruction. Expand **View Shot List** to review the full plan before committing to assembly.

**3. Start Video Assembly**

Click **🚀 Start Video Assembly**. Assembly runs in a background thread — the progress bar updates every 3 seconds. **Keep this browser tab open** while it runs.

Typical times:
- YouTube Short: 1–3 minutes
- Mid-length (3-8 min): 4–8 minutes
- Deep Dive (10+ min): 8–15 minutes

**4. Download**

When complete, click **📥 Download MP4**. The filename includes `_shorts` for vertical videos so you can tell them apart.

Use **🔄 Reset & Assemble Again** to regenerate with a fresh shot list.

---

## Content Mode Reference

### Film & Series Analysis

Primary visuals come from the official trailer (if downloaded) and TMDB. Character names from the generated script package are passed directly to cast photo lookups — no guessing. TMDB backdrops are sorted landscape-first so portrait images no longer cause the pillarbox-crop problem. All portrait images get a blurred darkened background fill instead of a hard crop.

Pexels queries are written to match the film's specific world and atmosphere — not genre labels. For example, `"lone astronaut deep space silence"` rather than `"sci-fi movie"`.

### Tech News & Investigative

No TMDB. Visuals come from Pexels (topic-specific cinematic B-roll) and Wikipedia (company logos, product images, person photos — validated to minimum 200×200px to filter out icons and wrong matches). Stat cards carry the visual weight: impact numbers, downtime hours, affected user counts.

### Educational Technology

No TMDB. Pexels B-roll anchored to the specific concept. Code cards (dark VS Code-style panels with basic syntax colour-coding) appear when the script references actual code. Chapter cards mark section transitions using the script outline generated in Tab 3.

---

## Video Assembly — Visual Types by Mode

| Type | Description | Film | Tech | Edu |
|---|---|:---:|:---:|:---:|
| `trailer_clip` | Official trailer segment (yt-dlp) | ✅ | — | — |
| `tmdb_poster` | Official film poster | ✅ | — | — |
| `tmdb_backdrop` | Cinematic scene still | ✅ | — | — |
| `tmdb_cast` | Actor headshot | ✅ | — | — |
| `pexels_broll` | Cinematic B-roll | ✅ | ✅ | ✅ |
| `wiki_image` | Wikipedia thumbnail (validated) | — | ✅ | ✅ |
| `text_card` | Styled text graphic | ✅ | ✅ | ✅ |
| `stat_card` | Large-number impact card | ✅ | ✅ | ✅ |
| `code_card` | VS Code-style code panel | — | — | ✅ |
| `chapter_card` | Act/section transition | ✅ | ✅ | ✅ |

If any image fetch fails, SudoVid falls back automatically: first a fresh Pexels search on the topic, then any unused TMDB backdrop, then a generated text card. The video always completes.

All still clips use a gentle Ken Burns zoom effect (alternating in/out) so the video never feels static.

---

## Shorts vs Long-Form

| | YouTube Short | Long-Form |
|---|---|---|
| Canvas | 1080 × 1920 (9:16) | 1280 × 720 (16:9) |
| Total duration | 30–58 seconds | Matches voiceover length |
| Clip count | 3–6 | 8–20 |
| Clip duration | 5–12 seconds | 15–35 seconds |
| Trailer clips | ✅ Priority source | ✅ 30–50% of clips |
| Chapter cards | ✗ Not used | ✅ Up to 3 |
| Portrait images | Fill frame naturally | Pillarbox with blurred bg |
| Landscape images | Letterbox with blurred bg | Centre crop to fill |
| Credit position | Top centre, pill background | Bottom left, small text |

---

## Credit Overlay

Every assembled video has a credit text overlaid on every frame. The default is `"Courtesy: {topic}"` but you can edit it freely in Tab 6 before assembly.

- **YouTube Shorts** — white text centred at the top of the frame, inside a semi-transparent rounded pill
- **Long-form** — small white text at the bottom left, no background, subtle opacity

The overlay is composited as a final pass over the entire assembled video — it does not affect individual clip generation.

---

## File Structure

```
sudovid/
├── app.py              # Full application — all logic in one file
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Cloud (ffmpeg)
└── README.md           # This file
```

---

## Known Limitations

- **Browser tab must stay open during video assembly.** The background thread continues if the tab is closed, but the progress display and download button will not reappear on refresh.
- **Trailer download requires network access.** On restricted networks, `yt-dlp` may fail silently — the app falls back to stills automatically.
- **Streamlit Cloud free tier** has limited CPU. A Deep Dive video can take 10–15 minutes to render. Consider upgrading to a paid tier for faster assembly.
- **Edge TTS** requires an internet connection and does not work offline.
- **Session state is lost on browser refresh.** Use **Reset All Steps** in the sidebar for a clean restart — do not use the browser refresh button.
- **TMDB cast photos** are portrait-oriented. On long-form videos they render with a blurred pillarbox background. On Shorts they fill the frame naturally.
- **YouTube Shorts script** is under 150 words. The video assembly produces a 30–58 second vertical video. If your voiceover runs longer, trim the script in Tab 3 before generating audio.
