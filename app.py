import streamlit as st
import google.generativeai as genai
import json
import time
import math
import asyncio
import threading
import tempfile
import textwrap
import requests
import os
import re
import numpy as np
import edge_tts
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArchiText",
    page_icon="✍️",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    :root {
        --primary: #2563eb;
        --bg-main: #f8fafc;
        --bg-card: #ffffff;
        --border: #e2e8f0;
        --text-main: #1e293b;
        --text-secondary: #64748b;
    }
    .stApp { background-color: var(--bg-main); color: var(--text-main); }
    p, span, label, .stMarkdown, h1, h2, h3, .stMetric label {
        color: var(--text-main) !important;
    }
    .stCaption { color: var(--text-secondary) !important; }
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"], .stSelectbox div {
        background-color: white !important;
        border: 1px solid var(--border) !important;
        color: var(--text-main) !important;
    }
    .stButton>button {
        background-color: var(--primary); color: white; border-radius: 8px;
        height: 3.5em; font-weight: 600; width: 100%; border: none;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stButton>button:hover { background-color: #1d4ed8; transform: translateY(-1px); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: transparent;
        color: var(--text-secondary); font-weight: 600;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
    }
    .metric-badge {
        background-color: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe;
        padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 0.9em;
    }
    .report-card {
        background-color: white; padding: 24px; border-radius: 12px;
        border: 1px solid var(--border); margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricValue"] { color: var(--primary) !important; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — MODES
# ─────────────────────────────────────────────────────────────────────────────
MODE_FILM = "Film & Series Analysis"
MODE_TECH = "Tech News & Investigative"
MODE_EDU  = "Educational Technology"


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO  (edge-tts)
# ─────────────────────────────────────────────────────────────────────────────

async def _tts_async(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        await communicate.save(f.name)
        return f.name

def generate_audio_sync(text, voice):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_tts_async(text, voice))
    except Exception as e:
        st.error(f"Audio Generation Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(api_key, prompt, system_instruction="", use_search=False, is_json=False):
    if not use_search:
        genai.configure(api_key=api_key)
        gen_config = {"response_mime_type": "application/json"} if is_json else None
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            generation_config=gen_config,
        )
        for delay in [1, 2, 4, 8, 16]:
            try:
                return model.generate_content(prompt).text
            except Exception as e:
                if delay == 16:
                    return f"Error: {str(e)}"
                time.sleep(delay)
    else:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "tools": [{"google_search": {}}],
        }
        for delay in [1, 2, 4, 8, 16]:
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                if "candidates" in result and result["candidates"]:
                    parts = result["candidates"][0]["content"]["parts"]
                    return "\n".join(p.get("text", "") for p in parts if "text" in p)
                return "Error: Unexpected response format"
            except requests.exceptions.RequestException as e:
                if delay == 16:
                    return (f"Error: {str(e)}\n"
                            f"Response: {e.response.text if e.response else 'none'}")
                time.sleep(delay)
            except Exception as e:
                if delay == 16:
                    return f"Error: {str(e)}"
                time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def perform_grounded_research(topic, mode, source_type, angle, length, api_key):
    system_instruction = (
        "You are an expert Research Assistant. Always search the web for current, "
        "accurate information. Your goal is to fact-check and find data that supports "
        "the Creator's Angle."
    )
    prompt = f"""
    TOPIC: {topic}
    SOURCE TYPE: {source_type}
    VIDEO LENGTH: {length}
    CREATOR'S ANGLE / DRAFT: {angle}

    TASK: Execute targeted web searches to gather factual context that specifically
    supports, verifies, or fills the gaps in the "CREATOR'S ANGLE".

    INSTRUCTIONS:
    1. Do not just summarize the topic. Actively look for data, recent news, or
       historical parallels that make the Creator's Angle stronger.
    2. If the Creator's Angle is missing specific facts (like exact dates, character
       names, technology versions, or company statements), find them.
    3. If the Video Length is "Deep Dive (10+ mins)", gather extensive details and
       multiple perspectives.
    4. Provide your findings as a factual briefing. Cite your sources with URLs.
    """
    return call_gemini(api_key, prompt, system_instruction, use_search=True)


def generate_script_package(mode, topic, research, angle, matrix,
                             source_type, length, api_key):
    personas = {
        MODE_FILM: "Master YouTube Film Critic. Focus on narrative, character arcs, and thematic depth.",
        MODE_TECH: "Investigative Tech YouTuber. Focus on clarity, impact, and engaging storytelling.",
        MODE_EDU:  "Senior Developer turned YouTuber. Explain things naturally, like a mentor talking to a junior.",
    }
    prompt = f"""
    TOPIC: {topic}
    SOURCE TYPE: {source_type}
    VIDEO LENGTH: {length}
    CREATOR'S DRAFT / UNIQUE ANGLE: {angle}
    SELECTED MATRIX (Tone/Style): {matrix}
    TARGETED RESEARCH: {research}

    TASK: You are a professional, conversational YouTube scriptwriter. Your goal is
    to refine the "CREATOR'S DRAFT" into a highly engaging, human-sounding script
    ready for voiceover.

    CRITICAL INSTRUCTIONS:
    1. LENGTH ADAPTATION: The target video length is '{length}'.
       - If it is a YouTube Short, make the script extremely punchy, fast-paced,
         and under 150 words total.
       - If it is Mid-length or Deep Dive, flesh out the arguments with natural pacing.
    2. HUMAN TONE: The script MUST sound like a real person talking to a camera.
       Use conversational phrasing, rhetorical questions, and natural transitions.
       AVOID robotic listicles.
    3. ANGLE-FIRST REFINEMENT: Preserve the creator's unique perspective. Only use
       the "TARGETED RESEARCH" to factually support their points. Do NOT dump all
       the research into the script.
    4. ALIGNMENT: Match the tone indicated in the "SELECTED MATRIX".
    5. ESCAPE CHARACTERS: Ensure ALL double quotes inside your script text are
       properly escaped (e.g., \\"Like this\\") so the JSON remains completely valid.

    JSON SCHEMA REQUIREMENTS:
    {{
      "thematic_resonance": {{ "real_world_event": "String", "explanation": "Detailed parallel based on angle" }},
      "character_matrix": [ {{ "name": "Name", "role": "Main/Side", "arc_score": 0, "ghost_vs_truth": "String" }} ],
      "technical_report": {{ "script": 0, "direction": 0, "editing": 0, "acting": 0 }},
      "viral_title": "String (Catchy YouTube Title)",
      "hook_script": "String (A punchy, conversational opening hook)",
      "full_script": {{
          "intro": "Conversational intro flowing from the hook.",
          "act1": "Conversational Act 1.",
          "act2": "Conversational Act 2.",
          "act3": "Conversational Act 3.",
          "outro": "Natural conclusion and call-to-action."
      }},
      "script_outline": ["Brief point 1", "Brief point 2", "Brief point 3"],
      "seo_metadata": {{ "description": "String", "tags": ["tag1", "tag2"] }}
    }}
    """
    result = call_gemini(api_key, prompt, personas.get(mode), is_json=True)
    try:
        clean = result.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        return {"error": f"Synthesis failed to return valid JSON. Error: {str(e)}", "raw": result}


def generate_youtube_bundle(api_key, script_text):
    prompt = f"""
    Analyze the following YouTube script and create a complete SEO and packaging bundle.

    SCRIPT:
    {script_text}

    JSON SCHEMA REQUIREMENTS:
    {{
        "viral_title": "String (A high-CTR, emotional, and catchy YouTube title)",
        "description": "String (A full YouTube description including a hook, summary, and placeholder for social links)",
        "tags": ["tag1", "tag2", "tag3", "etc (Generate 15 highly relevant SEO tags)"],
        "hashtags": ["#tag1", "#tag2", "#tag3 (Generate 3-5 highly relevant hashtags)"],
        "thumbnail_prompt": "String (A highly detailed, visual prompt for an AI image generator to create a catchy, high-contrast, professional YouTube thumbnail. Specify lighting, subjects, and mood.)"
    }}
    """
    result = call_gemini(api_key, prompt,
                         "You are a master YouTube strategist and SEO expert.",
                         is_json=True)
    try:
        clean = result.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        return {"error": f"Failed to generate bundle. Error: {str(e)}", "raw": result}


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO ASSEMBLY — CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_W, VIDEO_H  = 1280, 720
FPS               = 24
TMDB_BASE         = "https://api.themoviedb.org/3"
TMDB_IMG          = "https://image.tmdb.org/t/p/w1280"
PEXELS_BASE       = "https://api.pexels.com/v1/search"
WIKI_THUMB        = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
PROGRESS_FILE     = os.path.join(tempfile.gettempdir(), "sa_video_progress.json")

_DARK_BG  = (10, 12, 20)
_ACCENT   = (37,  99, 235)   # blue  — film
_TECH_ACC = (234, 88,  12)   # orange — tech
_EDU_ACC  = (22, 163,  74)   # green  — edu
_TEXT_HI  = (240, 245, 255)
_TEXT_LO  = (148, 163, 184)


def _write_progress(pct, msg, done=False, error=""):
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"pct": pct, "msg": msg, "done": done,
                       "error": error, "ts": time.time()}, f)
    except Exception:
        pass


def _read_progress():
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"pct": 0, "msg": "Starting…", "done": False, "error": ""}


def _font(size, bold=False):
    paths = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    )
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _mono_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE FETCH
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_array(url):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        w, h = img.size
        target = VIDEO_W / VIDEO_H
        if (w / h) > target:
            nw = int(h * target)
            img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w / target)
            img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
        return np.array(img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL IMAGE SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def _tmdb_lookup(topic, key):
    try:
        r = requests.get(f"{TMDB_BASE}/search/multi",
                         params={"api_key": key, "query": topic},
                         timeout=10).json()
        hits = r.get("results", [])
        if hits:
            return hits[0]["id"], hits[0].get("media_type", "movie")
    except Exception:
        pass
    return None, None


def _tmdb_images(media_id, media_type, key):
    try:
        r = requests.get(f"{TMDB_BASE}/{media_type}/{media_id}/images",
                         params={"api_key": key}, timeout=10).json()
        return {
            "backdrops": [TMDB_IMG + x["file_path"] for x in r.get("backdrops", [])[:15]],
            "posters":   [TMDB_IMG + x["file_path"] for x in r.get("posters",   [])[:3]],
        }
    except Exception:
        return {"backdrops": [], "posters": []}


def _tmdb_cast_photo(name, key):
    try:
        r = requests.get(f"{TMDB_BASE}/search/person",
                         params={"api_key": key, "query": name},
                         timeout=10).json()
        hits = r.get("results", [])
        if hits and hits[0].get("profile_path"):
            return TMDB_IMG + hits[0]["profile_path"]
    except Exception:
        pass
    return None


def _pexels_image(query, key, page=1):
    try:
        r = requests.get(
            PEXELS_BASE,
            headers={"Authorization": key},
            params={"query": query, "per_page": 5,
                    "page": page, "orientation": "landscape"},
            timeout=10,
        ).json()
        photos = r.get("photos", [])
        if photos:
            return photos[0]["src"]["large2x"]
    except Exception:
        pass
    return None


def _wiki_image(query):
    try:
        slug = query.strip().replace(" ", "_")
        r = requests.get(WIKI_THUMB.format(slug), timeout=10).json()
        return r.get("thumbnail", {}).get("source")
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CARD GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _base_card(accent):
    img  = Image.new("RGB", (VIDEO_W, VIDEO_H), _DARK_BG)
    draw = ImageDraw.Draw(img)
    for y in range(VIDEO_H):
        t = y / VIDEO_H
        s = int(10 + 18 * math.sin(math.pi * t))
        draw.line([(0, y), (VIDEO_W, y)], fill=(s, s + 2, s + 10))
    draw.rectangle([(0, 0), (6, VIDEO_H)], fill=accent)
    return img, draw


def _accent_for_mode(mode):
    return _ACCENT if mode == MODE_FILM else (_TECH_ACC if mode == MODE_TECH else _EDU_ACC)


def make_text_card(line1, line2="", label="", mode=MODE_FILM):
    img, draw = _base_card(_accent_for_mode(mode))
    cy = VIDEO_H // 2
    if label:
        draw.text((VIDEO_W // 2, cy - 100), label.upper(),
                  font=_font(22), fill=_accent_for_mode(mode), anchor="mm")
    wrapped = "\n".join(textwrap.wrap(line1, width=34))
    draw.text((VIDEO_W // 2, cy - (30 if line2 else 0)), wrapped,
              font=_font(58, bold=True), fill=_TEXT_HI, anchor="mm", align="center")
    if line2:
        draw.text((VIDEO_W // 2, cy + 70), line2,
                  font=_font(34), fill=_TEXT_LO, anchor="mm")
    return np.array(img)


def make_chapter_card(chapter_num, title, mode=MODE_FILM):
    accent = _accent_for_mode(mode)
    img, draw = _base_card(accent)
    draw.text((VIDEO_W - 60, VIDEO_H - 40), f"#{chapter_num}",
              font=_font(96, bold=True), fill=(*accent, 30), anchor="rb")
    draw.text((VIDEO_W // 2, VIDEO_H // 2 - 40), f"PART {chapter_num}",
              font=_font(24), fill=accent, anchor="mm")
    wrapped = "\n".join(textwrap.wrap(title, width=36))
    draw.text((VIDEO_W // 2, VIDEO_H // 2 + 30), wrapped,
              font=_font(52, bold=True), fill=_TEXT_HI, anchor="mm", align="center")
    return np.array(img)


def make_stat_card(stat, description, mode=MODE_TECH):
    accent = _accent_for_mode(mode)
    img, draw = _base_card(accent)
    draw.text((VIDEO_W // 2, VIDEO_H // 2 - 60), stat,
              font=_font(110, bold=True), fill=_TEXT_HI, anchor="mm")
    draw.text((VIDEO_W // 2, VIDEO_H // 2 + 70), description,
              font=_font(36), fill=_TEXT_LO, anchor="mm")
    return np.array(img)


def make_code_card(snippet, language=""):
    img  = Image.new("RGB", (VIDEO_W, VIDEO_H), (18, 20, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (VIDEO_W, 44)], fill=(30, 32, 44))
    for i, col in enumerate([(255, 95, 87), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([(18 + i * 22, 14), (32 + i * 22, 28)], fill=col)
    if language:
        draw.text((VIDEO_W - 20, 22), language.upper(),
                  font=_font(16), fill=(100, 116, 139), anchor="rm")
    mf    = _mono_font(26)
    lines = snippet.strip().split("\n")[:18]
    y     = 64
    for i, line in enumerate(lines):
        draw.text((50, y), str(i + 1), font=mf, fill=(64, 74, 94), anchor="rm")
        stripped = line.lstrip()
        if stripped.startswith(("#", "//")):
            col = (106, 153, 85)
        elif any(stripped.startswith(k) for k in (
            "def ", "class ", "import ", "from ", "return ",
            "const ", "let ", "var ", "function ", "async "
        )):
            col = (197, 134, 192)
        elif stripped.startswith(("'", '"', "`", "f'")):
            col = (206, 145, 120)
        else:
            col = _TEXT_HI
        draw.text((64, y), line, font=mf, fill=col)
        y += 34
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# KEN BURNS ZOOM
# ─────────────────────────────────────────────────────────────────────────────

def _zoom_clip(arr, duration, zoom_in=True):
    pil = Image.fromarray(arr)
    h0, w0 = arr.shape[:2]

    def make_frame(t):
        p     = t / max(duration, 0.001)
        scale = 1.0 + 0.04 * (p if zoom_in else (1 - p))
        nw, nh = int(w0 * scale), int(h0 * scale)
        res   = np.array(pil.resize((nw, nh), Image.BILINEAR))
        x0 = (nw - w0) // 2
        y0 = (nh - h0) // 2
        return res[y0:y0 + h0, x0:x0 + w0]

    return VideoClip(make_frame, duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# SHOT LIST PROMPTS  — one per mode
# ─────────────────────────────────────────────────────────────────────────────

def _shot_prompt_film(topic, source_type, matrix, character_names, script_outline, script_text):
    char_list = ", ".join(character_names) if character_names else "none"
    outline   = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    return f"""
You are a video editor planning a YouTube FILM / SERIES REVIEW video.

FILM TITLE: {topic}
SOURCE TYPE: {source_type}
TONE MATRIX: {matrix}
KNOWN CAST MEMBERS: {char_list}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

Produce a shot list. Divide the script into 8-20 timed segments (15-35s each).
For each segment pick ONE visual type:

  "tmdb_poster"   — official film poster (use for opening title card)
  "tmdb_backdrop" — cinematic scene still from TMDB
  "tmdb_cast"     — headshot of a named cast member (only names from KNOWN CAST MEMBERS)
  "pexels_broll"  — cinematic B-roll (mood, transitions, gaps)
  "text_card"     — styled card (RT score, box office, key review line, scene reference)
  "stat_card"     — big-number card ("$240M opening weekend", "8.4/10 IMDb")
  "chapter_card"  — act transition card from script outline (max 3 total)

Return ONLY a valid JSON array. No markdown. Schema per item:
{{
  "segment_index": 0, "type": "tmdb_poster", "duration_seconds": 10,
  "pexels_query": "", "cast_name": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "", "note": ""
}}

Rules:
- Only fill fields relevant to chosen type; leave others as empty string or 0.
- pexels_query: 3-5 words, only for pexels_broll.
- cast_name: exact name from KNOWN CAST MEMBERS, only for tmdb_cast.
- stat_value + stat_desc: only for stat_card.
- chapter_num + chapter_title: from script outline, only for chapter_card.
- Sum of duration_seconds ≈ word_count / 130 * 60 seconds.
- Mix: at least 2 tmdb types, 2 text/stat cards, 1 chapter_card.
"""


def _shot_prompt_tech(topic, matrix, script_outline, script_text):
    outline = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    return f"""
You are a video editor planning a YouTube TECH NEWS / INVESTIGATIVE video.

TOPIC: {topic}
CRITICALITY / SCOPE: {matrix}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

Produce a shot list. Divide the script into 8-20 timed segments (15-35s each).
For each segment pick ONE visual type:

  "pexels_broll"  — cinematic tech B-roll (servers, code screens, offices, data centres)
  "wiki_image"    — Wikipedia image for a company, product, or person mentioned
  "text_card"     — headline, key quote, timeline event, error message, policy statement
  "stat_card"     — big-number impact card ("8.5M users affected", "72hrs downtime")
  "chapter_card"  — section transition from script outline (max 3 total)

Return ONLY a valid JSON array. No markdown. Schema per item:
{{
  "segment_index": 0, "type": "pexels_broll", "duration_seconds": 20,
  "pexels_query": "", "wiki_title": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "", "note": ""
}}

Rules:
- pexels_query: 3-5 words, only for pexels_broll.
- wiki_title: exact Wikipedia article title, only for wiki_image.
- stat_value + stat_desc: only for stat_card.
- chapter_num + chapter_title: from script outline, only for chapter_card.
- Sum of duration_seconds ≈ word_count / 130 * 60 seconds.
- Mix: at least 40% pexels_broll, 2 stat_cards, 1 chapter_card.
"""


def _shot_prompt_edu(topic, matrix, script_outline, script_text):
    outline = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    return f"""
You are a video editor planning a YouTube EDUCATIONAL / TUTORIAL video.

TOPIC: {topic}
KNOWLEDGE LEVEL / STYLE: {matrix}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

Produce a shot list. Divide the script into 8-20 timed segments (15-35s each).
For each segment pick ONE visual type:

  "pexels_broll"  — cinematic concept B-roll matching the topic
  "wiki_image"    — Wikipedia image for a concept, person, or tool mentioned
  "text_card"     — definition, key term, or important takeaway
  "code_card"     — dark-themed code snippet (only when script references actual code)
  "stat_card"     — big-number fact ("O(n log n) complexity", "99.9% uptime")
  "chapter_card"  — section transition from script outline (max 3 total)

Return ONLY a valid JSON array. No markdown. Schema per item:
{{
  "segment_index": 0, "type": "pexels_broll", "duration_seconds": 20,
  "pexels_query": "", "wiki_title": "", "text_line1": "", "text_line2": "",
  "code_snippet": "", "code_language": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "", "note": ""
}}

Rules:
- pexels_query: 3-5 words, only for pexels_broll.
- wiki_title: exact Wikipedia article title, only for wiki_image.
- code_snippet: real code, max 12 lines, only for code_card.
- code_language: e.g. "Python", only for code_card.
- stat_value + stat_desc: only for stat_card.
- chapter_num + chapter_title: from script outline, only for chapter_card.
- Sum of duration_seconds ≈ word_count / 130 * 60 seconds.
- Mix: at least 30% pexels_broll, 2 text/stat cards, 1 chapter_card.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SHOT LIST GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_shot_list(api_key):
    mode           = st.session_state.get("mode_param", MODE_FILM)
    topic          = st.session_state.get("topic_param", "")
    matrix         = st.session_state.get("matrix_param", "")
    source_type    = st.session_state.get("source_param", "Original")
    script_text    = st.session_state.get("final_script_text", "")
    package        = st.session_state.get("package", {})
    script_outline = package.get("script_outline", [])

    if mode == MODE_FILM:
        char_names = [c["name"] for c in package.get("character_matrix", [])]
        prompt = _shot_prompt_film(topic, source_type, matrix,
                                   char_names, script_outline, script_text)
    elif mode == MODE_TECH:
        prompt = _shot_prompt_tech(topic, matrix, script_outline, script_text)
    else:
        prompt = _shot_prompt_edu(topic, matrix, script_outline, script_text)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={"response_mime_type": "application/json"},
    )
    for delay in [1, 2, 4, 8]:
        try:
            raw = model.generate_content(prompt).text
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception:
            time.sleep(delay)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# CLIP RESOLVER  — one shot dict → ndarray
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_clip(shot, mode, tmdb_cache, tmdb_key, pexels_key,
                  backdrop_idx, topic):
    stype = shot.get("type", "text_card")
    arr   = None

    # Film-specific
    if stype == "tmdb_poster":
        urls = tmdb_cache.get("posters", [])
        if urls:
            arr = _fetch_array(urls[0])

    elif stype == "tmdb_backdrop":
        urls = tmdb_cache.get("backdrops", [])
        if urls:
            arr = _fetch_array(urls[backdrop_idx[0] % len(urls)])
            backdrop_idx[0] += 1

    elif stype == "tmdb_cast":
        name = shot.get("cast_name", "")
        if name and tmdb_key:
            url = _tmdb_cast_photo(name, tmdb_key)
            if url:
                arr = _fetch_array(url)

    # Shared
    elif stype == "pexels_broll":
        query = shot.get("pexels_query", "cinematic abstract")
        if pexels_key:
            url = _pexels_image(query, pexels_key, page=(backdrop_idx[0] % 5) + 1)
            backdrop_idx[0] += 1
            if url:
                arr = _fetch_array(url)

    elif stype == "wiki_image":
        url = _wiki_image(shot.get("wiki_title", topic))
        if url:
            arr = _fetch_array(url)

    # Generated cards
    elif stype == "text_card":
        arr = make_text_card(
            shot.get("text_line1", shot.get("note", topic)),
            shot.get("text_line2", ""), mode=mode)

    elif stype == "stat_card":
        arr = make_stat_card(
            shot.get("stat_value", ""), shot.get("stat_desc", ""), mode=mode)

    elif stype == "code_card":
        arr = make_code_card(
            shot.get("code_snippet", "# No code provided"),
            shot.get("code_language", ""))

    elif stype == "chapter_card":
        arr = make_chapter_card(
            int(shot.get("chapter_num", 1) or 1),
            shot.get("chapter_title", shot.get("note", "")), mode=mode)

    # Fallback chain
    if arr is None:
        if pexels_key and stype in ("tmdb_poster", "tmdb_backdrop",
                                    "tmdb_cast", "wiki_image"):
            url = _pexels_image(topic + " cinematic", pexels_key)
            if url:
                arr = _fetch_array(url)
    if arr is None:
        arr = make_text_card(shot.get("note", topic), mode=mode)

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ASSEMBLY WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _assembly_worker(audio_path, shot_list, mode, topic,
                     tmdb_key, pexels_key, output_path):
    try:
        # Pre-fetch TMDB (film only)
        tmdb_cache = {"backdrops": [], "posters": []}
        if mode == MODE_FILM and tmdb_key:
            _write_progress(3, "Looking up film on TMDB…")
            mid, mtype = _tmdb_lookup(topic, tmdb_key)
            if mid:
                tmdb_cache = _tmdb_images(mid, mtype or "movie", tmdb_key)

        total        = len(shot_list)
        backdrop_idx = [0]
        clips        = []

        for i, shot in enumerate(shot_list):
            pct = 10 + int(72 * i / total)
            _write_progress(pct,
                f"Building clip {i+1}/{total} [{shot.get('type','?')}] — "
                f"{shot.get('note', '')}")

            duration = max(float(shot.get("duration_seconds", 12)), 5.0)
            zoom_in  = (i % 2 == 0)
            arr = _resolve_clip(shot, mode, tmdb_cache, tmdb_key,
                                pexels_key, backdrop_idx, topic)
            clips.append(_zoom_clip(arr, duration, zoom_in))

        _write_progress(84, "Stitching clips…")
        video = concatenate_videoclips(clips, method="compose")

        _write_progress(89, "Attaching voiceover…")
        audio = AudioFileClip(audio_path)

        # Match video length to audio
        if video.duration < audio.duration:
            extra  = audio.duration - video.duration + clips[-1].duration
            filler = _zoom_clip(clips[-1].get_frame(0), extra, zoom_in=False)
            clips[-1] = filler
            video = concatenate_videoclips(clips, method="compose")

        final = video.subclip(0, audio.duration).set_audio(audio)

        _write_progress(93, "Rendering MP4 (this takes ~60s)…")
        final.write_videofile(
            output_path, fps=FPS, codec="libx264",
            audio_codec="aac", preset="ultrafast",
            threads=2, logger=None,
        )
        _write_progress(100, "Done!", done=True)

    except Exception as e:
        _write_progress(0, "", done=True, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🖊️ ArchiText")
st.caption("AI-Powered Script, Voice & Video Engine")

# SIDEBAR
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        st.success("✓ Gemini API Key provided")
    else:
        st.warning("⚠️ Gemini API Key required")
    st.divider()
    if st.button("Reset All Steps"):
        st.session_state.clear()
        st.rerun()

# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Parameters",
    "2. Ground Research",
    "3. Generated Script",
    "4. Voiceover",
    "5. Content Bundle",
    "6. Video Assembly",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Step 1: Set Project Parameters & Angle")
    st.info("Define the scope, tone, and your unique perspective before the AI conducts its research.")

    topic = st.text_input("Topic or Title",
                           placeholder="e.g., The Night Manager Season 2, Crowdstrike Outage")

    col_a, col_b = st.columns(2)
    with col_a:
        active_mode = st.selectbox("Content Mode",
                                   [MODE_FILM, MODE_TECH, MODE_EDU])
    with col_b:
        video_length = st.selectbox("Target Video Length", [
            "YouTube Short (< 1 minute)",
            "Mid-length (3-8 mins)",
            "Deep Dive (10+ mins)",
        ])

    source_type = "Original"
    if active_mode == MODE_FILM:
        source_type = st.radio("Source Material",
                                ["Original", "Book", "Comic", "True Event", "Remake"],
                                horizontal=True)

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    matrix_data = {}
    if active_mode == MODE_FILM:
        c1, c2 = st.columns(2)
        with c1:
            matrix_data["Theory"]   = st.select_slider("Film Theory Focus",
                                        ["Formalist", "Psychological", "Auteur", "Montage"])
            matrix_data["Visuals"]  = st.select_slider("Visual Signature",
                                        ["Standard", "Stylized", "Iconic"])
        with c2:
            matrix_data["Fidelity"] = st.select_slider("Adaptation Fidelity",
                                        ["Loose", "Balanced", "Literal"])
            matrix_data["Tone"]     = st.selectbox("Narrative Tone",
                                        ["Conversational", "Melancholic", "Frantic", "Academic"])
    elif active_mode == MODE_TECH:
        matrix_data["Severity"] = st.select_slider("Criticality", ["Bug", "Outage", "Crisis"])
        matrix_data["Scope"]    = st.select_slider("User Impact",  ["Niche", "Widespread", "Global"])
    else:
        matrix_data["Complexity"] = st.select_slider("Knowledge Level",
                                      ["Junior", "Senior", "Architect"])
        matrix_data["Method"]     = st.select_slider("Pedagogical Style",
                                      ["Theory", "Mixed", "Practical"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### ✍️ Your Unique Angle (Draft)")
    st.caption("Upload your notes or type your perspective here. The AI will use this to guide its research in Step 2.")
    angle_file = st.file_uploader("Upload your rough draft or notes (.txt)", type=["txt"])
    angle_text = st.text_area("Or type your angle/rough draft here:", height=150,
                               placeholder="E.g., I think the main character's arc was ruined in episode 4 because…")

    if st.button("💾 Save Parameters & Proceed"):
        final_angle = angle_text.strip() or (
            angle_file.getvalue().decode("utf-8") if angle_file else ""
        )
        if not topic:
            st.error("⚠️ Please provide a Topic or Title.")
        elif not final_angle:
            st.error("⚠️ Please provide your Unique Angle.")
        else:
            st.session_state["topic_param"]  = topic
            st.session_state["mode_param"]   = active_mode
            st.session_state["length_param"] = video_length
            st.session_state["source_param"] = source_type
            st.session_state["matrix_param"] = json.dumps(matrix_data)
            st.session_state["angle_param"]  = final_angle
            st.success("✅ Parameters saved! Click the **'2. Ground Research'** tab.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GROUND RESEARCH
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Step 2: Targeted Intelligence Gathering")
    if "topic_param" not in st.session_state:
        st.info("Please complete Step 1 (Parameters) first and click 'Save Parameters'.")
    else:
        st.info(f"🌐 The AI will search the web to fact-check and support your angle on "
                f"**{st.session_state['topic_param']}**.")
        if st.button("🔍 Execute Targeted Background Research"):
            if not api_key:
                st.warning("Please provide a Gemini API Key in the sidebar.")
            else:
                with st.spinner("🌐 Actively searching the web…"):
                    st.session_state["research"] = perform_grounded_research(
                        topic       = st.session_state["topic_param"],
                        mode        = st.session_state["mode_param"],
                        source_type = st.session_state["source_param"],
                        angle       = st.session_state["angle_param"],
                        length      = st.session_state["length_param"],
                        api_key     = api_key,
                    )
        if "research" in st.session_state:
            st.success("✅ Targeted Research Complete")
            with st.expander("View Factual Briefing", expanded=False):
                st.markdown(st.session_state["research"])
            st.success("🎉 **Step 2 Complete!** Click the **'3. Generated Script'** tab.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GENERATED SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Step 3: Script Generation & Editing")
    if "research" not in st.session_state:
        st.info("Complete Step 2 (Ground Research) first.")
    else:
        if st.button("🚀 Architect Refined Script"):
            with st.spinner(f"Synthesizing your script for a {st.session_state['length_param']}…"):
                st.session_state["package"] = generate_script_package(
                    mode        = st.session_state["mode_param"],
                    topic       = st.session_state["topic_param"],
                    research    = st.session_state["research"],
                    angle       = st.session_state["angle_param"],
                    matrix      = st.session_state["matrix_param"],
                    source_type = st.session_state["source_param"],
                    length      = st.session_state["length_param"],
                    api_key     = api_key,
                )

        if "package" in st.session_state:
            p = st.session_state["package"]
            if "error" in p:
                st.error(p["error"])
                with st.expander("View Raw Output (Debugging)"):
                    st.text(p.get("raw"))
            else:
                st.success(f"### {p.get('viral_title')}")

                with st.expander("📊 View Script Architecture Details", expanded=False):
                    st.markdown("#### 🌍 Thematic Resonance")
                    st.warning(f"**Analogous Event:** {p.get('thematic_resonance', {}).get('real_world_event')}")
                    st.write(p.get("thematic_resonance", {}).get("explanation"))
                    if st.session_state["mode_param"] == MODE_FILM:
                        for char in p.get("character_matrix", []):
                            st.markdown(
                                f"**{char['name']}** "
                                f"<span class='metric-badge'>{char['arc_score']}/10</span>",
                                unsafe_allow_html=True,
                            )

                st.markdown("### 📝 Conversational Script Editor")
                st.info("💡 Edit below as you want it spoken. Use commas or --- for natural pauses.")

                fs = p.get("full_script", {})
                default_text = "\n\n".join(filter(None, [
                    p.get("hook_script", ""),
                    fs.get("intro", ""),
                    fs.get("act1",  ""),
                    fs.get("act2",  ""),
                    fs.get("act3",  ""),
                    fs.get("outro", ""),
                ]))
                st.session_state["final_script_text"] = st.text_area(
                    "Final Polish:", value=default_text.strip(), height=400)

                st.download_button(
                    label="📥 Download Text Script",
                    data=st.session_state["final_script_text"],
                    file_name=f"{p.get('viral_title','script').replace(' ','_').lower()}.txt",
                    mime="text/plain",
                )
                st.success("🎉 **Step 3 Complete!** Click the **'4. Voiceover'** tab.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — VOICEOVER
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Step 4: AI Voiceover Studio")
    st.info("Turn your finalized script or a custom uploaded file into professional audio.")

    st.markdown("### 🎙️ Voice Settings")
    voice_option = st.selectbox("Select Narrator (US English)", [
        ("en-US-ChristopherNeural", "Christopher (Male - Deep/Professional)"),
        ("en-US-GuyNeural",         "Guy (Male - Natural/Conversational)"),
        ("en-US-EricNeural",        "Eric (Male - Casual)"),
        ("en-US-RogerNeural",       "Roger (Male - Confident)"),
        ("en-US-SteffanNeural",     "Steffan (Male - Expressive)"),
        ("en-US-AndrewNeural",      "Andrew (Male - Warm)"),
        ("en-US-BrianNeural",       "Brian (Male - Crisp/News)"),
        ("en-US-AriaNeural",        "Aria (Female - Clear)"),
        ("en-US-JennyNeural",       "Jenny (Female - Conversational)"),
        ("en-US-MichelleNeural",    "Michelle (Female - Bright)"),
        ("en-US-EmmaNeural",        "Emma (Female - Friendly)"),
        ("en-US-AvaNeural",         "Ava (Female - Engaging)"),
    ], format_func=lambda x: x[1])

    st.markdown("---")
    source_mode = st.radio("Choose Text Source for Voiceover:",
                            ["Use Generated Script (from Tab 3)",
                             "Upload Custom Text File (.txt)"])

    text_to_synthesize = ""
    if source_mode == "Use Generated Script (from Tab 3)":
        text_to_synthesize = st.session_state.get("final_script_text", "")
        if not text_to_synthesize:
            st.warning("⚠️ No generated script found. Complete Steps 1-3 first, "
                       "or select 'Upload Custom Text File'.")
    else:
        uploaded_file = st.file_uploader("Upload a .txt file for Voiceover",
                                          type=["txt"], key="voice_upload")
        if uploaded_file:
            text_to_synthesize = uploaded_file.getvalue().decode("utf-8")
            st.success("File uploaded successfully!")

    st.markdown("---")
    st.markdown("### Preview Text for Audio Generation")
    st.session_state["tab4_audio_text"] = st.text_area(
        "This exact text will be sent to the AI Voice:",
        value=text_to_synthesize, height=250)

    if st.button("🔊 Generate Voiceover"):
        if not st.session_state["tab4_audio_text"].strip():
            st.error("Text box is empty. Please provide text to generate audio.")
        else:
            with st.spinner(f"Synthesising audio with {voice_option[1]} "
                            f"(this may take 10-20 seconds)…"):
                audio_path = generate_audio_sync(
                    st.session_state["tab4_audio_text"], voice_option[0])
                if audio_path:
                    # ── save path for Tab 6 ──
                    st.session_state["last_audio_path"] = audio_path
                    st.success("✅ Audio generated successfully!")
                    st.audio(audio_path, format="audio/mp3")
                    with open(audio_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Audio File (.mp3)",
                            data=f,
                            file_name=f"{st.session_state.get('topic_param','voiceover').replace(' ','_').lower()}_voiceover.mp3",
                            mime="audio/mp3",
                        )
                else:
                    st.error("Failed to generate audio. Check your internet connection.")
            st.success("🎉 **Step 4 Complete!** Click the **'5. Content Bundle'** tab.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CONTENT BUNDLE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Step 5: YouTube Content Bundle")
    st.info("Package your final script with a viral title, SEO description, tags, and thumbnail concept.")

    bundle_source = st.radio("Select the script text to use as the foundation:",
                              ["Use 'Generated Script' (from Tab 3)",
                               "Use 'Final Audio Text' (from Tab 4)"])

    if st.button("📦 Generate Content Bundle"):
        target_text = (
            st.session_state.get("final_script_text", "")
            if bundle_source == "Use 'Generated Script' (from Tab 3)"
            else st.session_state.get("tab4_audio_text", "")
        )
        if not api_key:
            st.error("⚠️ Gemini API Key required.")
        elif not target_text.strip():
            st.error("⚠️ Target text is empty. Generate or upload a script first.")
        else:
            with st.spinner("Analysing script and generating YouTube metadata…"):
                st.session_state["yt_bundle"] = generate_youtube_bundle(api_key, target_text)

    if "yt_bundle" in st.session_state:
        bundle = st.session_state["yt_bundle"]
        if "error" in bundle:
            st.error(bundle["error"])
        else:
            st.success("✅ YouTube Bundle Generated!")
            st.markdown("### 📝 YouTube Metadata")
            st.text_input("**Viral Title**",       value=bundle.get("viral_title",  ""))
            st.text_area( "**Description**",        value=bundle.get("description",  ""), height=200)
            col_tags, col_hashes = st.columns(2)
            with col_tags:
                st.text_area("**Tags** (comma separated)",
                             value=", ".join(bundle.get("tags", [])), height=100)
            with col_hashes:
                st.text_area("**Hashtags**",
                             value=" ".join(bundle.get("hashtags", [])), height=100)
            st.markdown("---")
            st.markdown("### 🎨 AI Thumbnail Prompt")
            st.caption("Copy into Midjourney, DALL-E, or Canva to create your thumbnail.")
            st.text_area("Suggested Image Prompt:",
                         value=bundle.get("thumbnail_prompt", ""), height=100)
            st.success("🎉 **Step 5 Complete!** Click the **'6. Video Assembly'** tab.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — VIDEO ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Step 6: Video Assembly")

    mode  = st.session_state.get("mode_param", "")
    topic_val = st.session_state.get("topic_param", "")

    mode_labels = {MODE_FILM: "🎬 Film & Series",
                   MODE_TECH: "🔍 Tech / Investigative",
                   MODE_EDU:  "📚 Educational"}
    if mode:
        st.info(f"Mode: **{mode_labels.get(mode, mode)}** — Topic: **{topic_val}**")

    # API Keys
    with st.expander("🔑 Image API Keys", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tmdb_key = st.text_input(
                "TMDB API Key"
                + (" *(required for film stills)*"
                   if mode == MODE_FILM
                   else " *(not used for this mode)*"),
                type="password", key="tmdb_key_input",
            )
        with col2:
            pexels_key = st.text_input(
                "Pexels API Key *(required — free at pexels.com/api)*",
                type="password", key="pexels_key_input",
            )

    st.markdown("---")

    # Pre-flight checks
    audio_path_v = st.session_state.get("last_audio_path", "")
    script_text_v = st.session_state.get("final_script_text", "")

    missing = []
    if not audio_path_v or not os.path.exists(audio_path_v):
        missing.append("✗ No audio file — complete Step 4 (Voiceover) first")
    if not script_text_v:
        missing.append("✗ No script — complete Step 3 first")
    if not topic_val:
        missing.append("✗ No topic — complete Step 1 first")
    if not api_key:
        missing.append("✗ Gemini API Key required (sidebar)")
    if not pexels_key:
        missing.append("✗ Pexels API Key required above")
    if mode == MODE_FILM and not tmdb_key:
        missing.append("✗ TMDB API Key required for Film & Series mode")

    if missing:
        for m in missing:
            st.warning(m)
    else:
        # Shot list generation
        if st.button("🎬 Generate Shot List"):
            with st.spinner("Gemini is planning your shot list…"):
                st.session_state["shot_list"] = generate_shot_list(api_key)

        if "shot_list" in st.session_state:
            sl        = st.session_state["shot_list"]
            total_dur = sum(s.get("duration_seconds", 12) for s in sl)
            st.success(f"Shot list ready — **{len(sl)} clips**, ~**{int(total_dur)}s** total")

            type_icons = {
                "tmdb_poster":  "🖼️", "tmdb_backdrop": "🎞️",
                "tmdb_cast":    "👤", "pexels_broll":  "🌆",
                "wiki_image":   "📖", "text_card":     "📝",
                "stat_card":    "📊", "code_card":     "💻",
                "chapter_card": "📌",
            }
            with st.expander("📋 View Shot List", expanded=False):
                for s in sl:
                    icon   = type_icons.get(s.get("type", ""), "▪️")
                    detail = (s.get("pexels_query") or s.get("cast_name") or
                              s.get("wiki_title")   or s.get("text_line1") or
                              s.get("stat_value")   or s.get("chapter_title") or
                              s.get("note", ""))
                    st.markdown(
                        f"`{s.get('segment_index',0):02d}` {icon} "
                        f"**{s.get('type','')}** — "
                        f"{s.get('duration_seconds','?')}s — _{detail}_"
                    )

            st.markdown("---")

            output_path = os.path.join(tempfile.gettempdir(), "sa_output_video.mp4")

            if "assembly_running" not in st.session_state:
                st.session_state["assembly_running"] = False

            if not st.session_state["assembly_running"]:
                if st.button("🚀 Start Video Assembly"):
                    _write_progress(0, "Initialising…")
                    st.session_state["assembly_running"] = True
                    st.session_state["assembly_output"]  = output_path

                    t = threading.Thread(
                        target=_assembly_worker,
                        kwargs=dict(
                            audio_path  = audio_path_v,
                            shot_list   = sl,
                            mode        = mode,
                            topic       = topic_val,
                            tmdb_key    = tmdb_key,
                            pexels_key  = pexels_key,
                            output_path = output_path,
                        ),
                        daemon=True,
                    )
                    t.start()
                    st.rerun()

            if st.session_state.get("assembly_running"):
                prog = _read_progress()

                if prog.get("error"):
                    st.error(f"Assembly failed: {prog['error']}")
                    st.session_state["assembly_running"] = False

                elif prog.get("done") and prog.get("pct", 0) == 100:
                    st.session_state["assembly_running"] = False
                    st.success("✅ Video assembled successfully!")
                    out = st.session_state.get("assembly_output", output_path)
                    if os.path.exists(out):
                        with open(out, "rb") as f:
                            st.download_button(
                                label="📥 Download MP4",
                                data=f,
                                file_name=f"{topic_val.replace(' ','_').lower()}_video.mp4",
                                mime="video/mp4",
                            )
                    if st.button("🔄 Reset & Assemble Again"):
                        for k in ("shot_list", "assembly_running", "assembly_output"):
                            st.session_state.pop(k, None)
                        st.rerun()

                else:
                    pct = prog.get("pct", 0)
                    msg = prog.get("msg", "Working…")
                    st.progress(pct / 100, text=f"{msg} ({pct}%)")
                    st.caption("⏳ Keep this tab open. Closing or idling will lose progress.")
                    time.sleep(3)
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("ArchiText v1.0 | AI-Powered Script, Voice & Video Engine")
