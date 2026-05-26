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
import bisect
import numpy as np
import edge_tts
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Monkey-patch to fix MoviePy 1.0.3 compatibility with Pillow 10+
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

from moviepy.editor import AudioFileClip, concatenate_videoclips, VideoFileClip
from moviepy.video.VideoClip import VideoClip

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SudoVid",
    page_icon="🎬",
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
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODE_FILM = "Film & Series Analysis"
MODE_TECH = "Tech News & Investigative"
MODE_EDU  = "Educational Technology"

# Canvas sizes
CANVAS_LANDSCAPE = (1280, 720)   # 16:9  long-form
CANVAS_SHORTS    = (1080, 1920)  # 9:16  Shorts

FPS             = 24
WIKI_THUMB      = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
OPENVERSE_BASE  = "https://api.openverse.org/v1/images/"
PROGRESS_FILE   = os.path.join(tempfile.gettempdir(), "sa_video_progress.json")

_DARK_BG  = (10, 12, 20)
_ACCENT   = (37,  99, 235)
_TECH_ACC = (234, 88,  12)
_EDU_ACC  = (22, 163,  74)
_TEXT_HI  = (240, 245, 255)
_TEXT_LO  = (148, 163, 184)

def _canvas(is_shorts: bool) -> tuple:
    return CANVAS_SHORTS if is_shorts else CANVAS_LANDSCAPE


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
# SCRIPT GENERATION
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
# PROGRESS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# FONT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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
# IMAGE FIT  — handles all aspect ratio combinations without distortion
# ─────────────────────────────────────────────────────────────────────────────

def _fit_to_canvas(img_pil: Image.Image, cw: int, ch: int) -> np.ndarray:
    sw, sh = img_pil.size
    target_ratio = cw / ch
    src_ratio    = sw / sh

    both_landscape = src_ratio >= 1.0 and target_ratio >= 1.0
    both_portrait  = src_ratio <  1.0 and target_ratio <  1.0

    if both_landscape or both_portrait:
        scale = max(cw / sw, ch / sh)
        nw, nh = int(sw * scale), int(sh * scale)
        resized = img_pil.resize((nw, nh), Image.LANCZOS)
        x0 = (nw - cw) // 2
        y0 = (nh - ch) // 2
        return np.array(resized.crop((x0, y0, x0 + cw, y0 + ch)))
    else:
        scale = min(cw / sw, ch / sh)
        nw, nh = int(sw * scale), int(sh * scale)
        resized = img_pil.resize((nw, nh), Image.LANCZOS)

        bg_scale = max(cw / sw, ch / sh)
        bg_w = max(int(sw * bg_scale), cw)
        bg_h = max(int(sh * bg_scale), ch)
        bg = img_pil.resize((bg_w, bg_h), Image.LANCZOS)
        bx = (bg_w - cw) // 2
        by = (bg_h - ch) // 2
        bg = bg.crop((bx, by, bx + cw, by + ch))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=22))
        bg_arr = (np.array(bg) * 0.30).astype(np.uint8)
        canvas = Image.fromarray(bg_arr)

        px = (cw - nw) // 2
        py = (ch - nh) // 2
        canvas.paste(resized, (px, py))
        return np.array(canvas)


def _fetch_array(url: str, cw: int, ch: int) -> np.ndarray | None:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if img.width < 100 or img.height < 100:
            return None
        return _fit_to_canvas(img, cw, ch)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL IMAGE SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def _openverse_image(query: str, page: int = 1) -> str | None:
    try:
        r = requests.get(
            OPENVERSE_BASE,
            params={
                "q":             query,
                "license_type":  "commercial",
                "page_size":     5,
                "page":          page,
            },
            headers={"User-Agent": "SudoVid/1.0"},
            timeout=12,
        ).json()
        results = r.get("results", [])
        if results:
            return results[0].get("url")
    except Exception:
        pass
    return None


def _wiki_image(query: str) -> str | None:
    try:
        slug = query.strip().replace(" ", "_")
        r = requests.get(WIKI_THUMB.format(slug), timeout=10).json()
        thumb = r.get("thumbnail", {})
        url = thumb.get("source")
        w   = thumb.get("width", 0)
        h   = thumb.get("height", 0)
        if url and w >= 200 and h >= 200:
            return url
    except Exception:
        pass
    return None


def _wiki_image_multi(queries: list[str]) -> str | None:
    for q in queries:
        url = _wiki_image(q)
        if url:
            return url
    return None


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL TRAILER DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────

def _download_custom_video(video_url: str, out_path: str) -> bool:
    """Downloads a video from almost any site (IMDb, YouTube, etc.) using yt-dlp."""
    try:
        import yt_dlp
        
        # Kill the phantom cache: force delete the old file
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except: pass
            
        opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
            "outtmpl":             out_path,
            "quiet":               True,
            "no_warnings":         True,
            "merge_output_format": "mp4",
            "overwrites":          True,  # Force overwrite
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url])
            
        return os.path.exists(out_path) and os.path.getsize(out_path) > 100_000
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CREDIT OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def _add_credit_overlay(frame_arr: np.ndarray,
                         credit_text: str,
                         is_shorts: bool,
                         cw: int, ch: int) -> np.ndarray:
    if not credit_text:
        return frame_arr
    img  = Image.fromarray(frame_arr.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(18, ch // 58)
    font = _font(font_size)

    bbox  = draw.textbbox((0, 0), credit_text, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]
    pad_x, pad_y = 14, 7

    if is_shorts:
        tx = (cw - tw) // 2
        ty = int(ch * 0.028)
        draw.rounded_rectangle(
            [tx - pad_x, ty - pad_y, tx + tw + pad_x, ty + th + pad_y],
            radius=10, fill=(0, 0, 0, 145))
        draw.text((tx, ty), credit_text, font=font, fill=(255, 255, 255, 230))
    else:
        tx = int(cw * 0.015)
        ty = int(ch * 0.938)
        draw.text((tx, ty), credit_text, font=font, fill=(255, 255, 255, 175))

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(composited)


# ─────────────────────────────────────────────────────────────────────────────
# CARD GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _accent_for_mode(mode):
    return _ACCENT if mode == MODE_FILM else (_TECH_ACC if mode == MODE_TECH else _EDU_ACC)


def _base_card(accent, cw, ch):
    img  = Image.new("RGB", (cw, ch), _DARK_BG)
    draw = ImageDraw.Draw(img)
    for y in range(ch):
        t = y / ch
        s = int(10 + 18 * math.sin(math.pi * t))
        draw.line([(0, y), (cw, y)], fill=(s, s + 2, s + 10))
    draw.rectangle([(0, 0), (6, ch)], fill=accent)
    return img, draw


def make_text_card(line1, line2="", label="", mode=MODE_FILM, cw=1280, ch=720):
    accent = _accent_for_mode(mode)
    img, draw = _base_card(accent, cw, ch)
    cy = ch // 2
    
    # Scale text by width to prevent cutoffs on Shorts
    scale = cw / 1280 
    
    if label:
        draw.text((cw // 2, cy - int(100 * scale)), label.upper(),
                  font=_font(max(16, int(22 * scale))), fill=accent, anchor="mm")
    wrapped = "\n".join(textwrap.wrap(line1, width=max(20, int(34 * (cw / 1280)))))
    draw.text((cw // 2, cy - (int(30 * scale) if line2 else 0)), wrapped,
              font=_font(max(28, int(58 * scale)), bold=True),
              fill=_TEXT_HI, anchor="mm", align="center")
    if line2:
        draw.text((cw // 2, cy + int(70 * scale)), line2,
                  font=_font(max(18, int(34 * scale))), fill=_TEXT_LO, anchor="mm")
    return np.array(img)


def make_chapter_card(chapter_num, title, mode=MODE_FILM, cw=1280, ch=720):
    accent = _accent_for_mode(mode)
    img, draw = _base_card(accent, cw, ch)
    
    scale = cw / 1280 
    
    draw.text((cw - int(60 * scale), ch - int(40 * scale)), f"#{chapter_num}",
              font=_font(max(48, int(96 * scale)), bold=True),
              fill=(*accent, 30), anchor="rb")
    draw.text((cw // 2, ch // 2 - int(40 * scale)), f"PART {chapter_num}",
              font=_font(max(14, int(24 * scale))), fill=accent, anchor="mm")
    wrapped = "\n".join(textwrap.wrap(title, width=max(20, int(36 * (cw / 1280)))))
    draw.text((cw // 2, ch // 2 + int(30 * scale)), wrapped,
              font=_font(max(26, int(52 * scale)), bold=True),
              fill=_TEXT_HI, anchor="mm", align="center")
    return np.array(img)


def make_stat_card(stat, description, mode=MODE_TECH, cw=1280, ch=720):
    accent = _accent_for_mode(mode)
    img, draw = _base_card(accent, cw, ch)
    
    scale = cw / 1280 
    
    draw.text((cw // 2, ch // 2 - int(60 * scale)), stat,
              font=_font(max(48, int(110 * scale)), bold=True),
              fill=_TEXT_HI, anchor="mm")
    draw.text((cw // 2, ch // 2 + int(70 * scale)), description,
              font=_font(max(18, int(36 * scale))), fill=_TEXT_LO, anchor="mm")
    return np.array(img)


def make_code_card(snippet, language="", cw=1280, ch=720):
    img  = Image.new("RGB", (cw, ch), (18, 20, 30))
    draw = ImageDraw.Draw(img)
    
    scale = cw / 1280 
    
    header_h = int(44 * scale)
    draw.rectangle([(0, 0), (cw, header_h)], fill=(30, 32, 44))
    dot_r = int(7 * scale)
    for i, col in enumerate([(255, 95, 87), (255, 189, 46), (39, 201, 63)]):
        cx_ = int((18 + i * 22) * scale)
        cy_ = int(22 * scale)
        draw.ellipse([cx_ - dot_r, cy_ - dot_r, cx_ + dot_r, cy_ + dot_r], fill=col)
    if language:
        draw.text((cw - int(20 * scale), int(22 * scale)), language.upper(),
                  font=_font(max(12, int(16 * scale))), fill=(100, 116, 139), anchor="rm")
    line_h = int(34 * scale)
    mf     = _mono_font(max(14, int(26 * scale)))
    lines  = snippet.strip().split("\n")[:18]
    y      = header_h + int(10 * scale)
    for i, line in enumerate(lines):
        draw.text((int(50 * scale), y), str(i + 1), font=mf,
                  fill=(64, 74, 94), anchor="rm")
        stripped = line.lstrip()
        if stripped.startswith(("#", "//")):               col = (106, 153, 85)
        elif any(stripped.startswith(k) for k in (
            "def ", "class ", "import ", "from ", "return ",
            "const ", "let ", "var ", "function ", "async ")): col = (197, 134, 192)
        elif stripped.startswith(("'", '"', "`", "f'")):   col = (206, 145, 120)
        else:                                               col = _TEXT_HI
        draw.text((int(64 * scale), y), line, font=mf, fill=col)
        y += line_h
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# KEN BURNS ZOOM
# ─────────────────────────────────────────────────────────────────────────────

def _zoom_clip(arr: np.ndarray, duration: float, zoom_in: bool = True):
    pil = Image.fromarray(arr.astype(np.uint8))
    h0, w0 = arr.shape[:2]

    def make_frame(t):
        p     = t / max(duration, 0.001)
        scale = 1.0 + 0.04 * (p if zoom_in else (1 - p))
        nw, nh = int(w0 * scale), int(h0 * scale)
        res    = np.array(pil.resize((nw, nh), Image.BILINEAR))
        x0_    = (nw - w0) // 2
        y0_    = (nh - h0) // 2
        return res[y0_:y0_ + h0, x0_:x0_ + w0].astype(np.float64)

    return VideoClip(make_frame, duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# SHOT LIST PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def _shot_prompt_film(topic, source_type, matrix, character_names,
                      script_outline, script_text,
                      has_trailer: bool, is_shorts: bool):
    char_list = ", ".join(character_names) if character_names else "none"
    outline   = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))

    if is_shorts:
        trailer_note = (
            'TRAILER AVAILABLE: Yes — prefer "trailer_clip" for all action/plot moments.'
            if has_trailer else
            'TRAILER AVAILABLE: No — use tmdb_backdrop and tmdb_poster as primary visuals.'
        )
        return f"""
You are a video editor cutting a YouTube SHORT (vertical 9:16, under 60 seconds).

FILM TITLE: {topic}
{trailer_note}
KNOWN CAST: {char_list}
SCRIPT: {script_text}

Produce a shot list of EXACTLY 3-6 segments, each 5-12 seconds.

Visual types allowed for Shorts:
  "trailer_clip"  — segment from the official trailer (PREFERRED if trailer available)
  "tmdb_poster"   — official film poster
  "tmdb_backdrop" — cinematic scene still
  "text_card"     — punchy one-liner card (max 1 in the whole list)

Return ONLY a valid JSON array. Schema per item:
{{
  "segment_index": 0, "type": "trailer_clip", "duration_seconds": 8,
  "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}

Rules:
- trailer_timestamp: approximate second in the trailer to cut from (spread across trailer).
- If trailer available: at least 60% of clips must be trailer_clip.
- If trailer not available: use tmdb_backdrop for all visual clips.
- Sum of duration_seconds must be between 30 and 58 seconds total.
- DO NOT use chapter_card, stat_card, code_card, or pexels_broll for Shorts.
"""
    else:
        trailer_note = (
            'TRAILER AVAILABLE: Yes — use "trailer_clip" for high-energy action/plot moments.'
            if has_trailer else
            'TRAILER AVAILABLE: No — use tmdb_backdrop and pexels_broll for visual coverage.'
        )
        return f"""
You are a video editor planning a YouTube FILM / SERIES REVIEW video.

FILM TITLE: {topic}
SOURCE TYPE: {source_type}
TONE MATRIX: {matrix}
{trailer_note}
KNOWN CAST MEMBERS: {char_list}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

Produce a shot list of 8-20 segments (15-35s each).
For each segment pick ONE visual type:

  "trailer_clip"  — segment from the official trailer (use for plot/action/emotional peaks)
  "tmdb_poster"   — official film poster (use for opening title card only)
  "tmdb_backdrop" — cinematic scene still (use for analytical/reflective moments)
  "tmdb_cast"     — headshot of a named cast member (only names from KNOWN CAST MEMBERS)
  "pexels_broll"  — cinematic B-roll SPECIFIC to this film's world:
                    queries must reference the film's setting, tone, or themes
                    (e.g. "lone astronaut deep space silence" not "sci-fi movie")
  "text_card"     — styled card (RT score, box office, key review line)
  "stat_card"     — big-number card ("$240M opening weekend", "8.4/10 IMDb")
  "chapter_card"  — act transition card from script outline (max 3 total)

Return ONLY a valid JSON array. No markdown. Schema per item:
{{
  "segment_index": 0, "type": "trailer_clip", "duration_seconds": 12,
  "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}

Rules:
- trailer_timestamp: spread timestamps across full trailer (avoid first 2s and last 5s).
- pexels_query: 4-6 words SPECIFIC to this film's world and atmosphere, NOT genre labels.
  Good: "lone astronaut deep space silence"  Bad: "sci-fi movie space"
- cast_name: exact name from KNOWN CAST MEMBERS only.
- If trailer available: use trailer_clip for 30-50% of clips.
- If trailer not available: skip trailer_clip entirely.
- Mix: at least 2 tmdb types, 2 text/stat cards, 1 chapter_card.
- Sum of duration_seconds ≈ word_count / 130 * 60 seconds.
"""


def _shot_prompt_tech(topic, matrix, script_outline, script_text, is_shorts):
    outline = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    if is_shorts:
        return f"""
You are a video editor cutting a YouTube SHORT (9:16, under 60 seconds).

TOPIC: {topic}
SCRIPT: {script_text}

Produce 3-6 segments, each 5-12 seconds.
Types allowed: "pexels_broll", "text_card", "stat_card" (max 1 each of text/stat).

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_broll", "duration_seconds": 8,
  "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules:
- pexels_query: 4-6 specific words (e.g. "server room crash red alert", not "technology")
- Sum 30-58 seconds total.
"""
    else:
        return f"""
You are a video editor planning a YouTube TECH NEWS / INVESTIGATIVE video.

TOPIC: {topic}
CRITICALITY / SCOPE: {matrix}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

Produce 8-20 segments (15-35s each). Types:

  "pexels_broll"  — cinematic tech B-roll SPECIFIC to this incident/product
                    (e.g. "crowdstrike blue screen windows crash", not "computer error")
  "wiki_image"    — Wikipedia image for a company, product, or person mentioned
  "text_card"     — headline, key quote, timeline event, error message
  "stat_card"     — big-number impact card
  "chapter_card"  — section transition (max 3 total)

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_broll", "duration_seconds": 20,
  "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules:
- pexels_query: 4-6 specific words tied to this exact topic, NOT generic tech labels.
- wiki_title: exact Wikipedia article title.
- Mix: at least 40% pexels_broll, 2 stat_cards, 1 chapter_card.
- Sum ≈ word_count / 130 * 60 seconds.
"""


def _shot_prompt_edu(topic, matrix, script_outline, script_text, is_shorts):
    outline = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    if is_shorts:
        return f"""
You are a video editor cutting a YouTube SHORT (9:16, under 60 seconds).

TOPIC: {topic}
SCRIPT: {script_text}

Produce 3-6 segments, 5-12 seconds each.
Types: "pexels_broll", "text_card", "code_card" (max 1 code_card).

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_broll", "duration_seconds": 8,
  "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules: pexels_query 4-6 specific words. Sum 30-58s total.
"""
    else:
        return f"""
You are a video editor planning a YouTube EDUCATIONAL / TUTORIAL video.

TOPIC: {topic}
KNOWLEDGE LEVEL / STYLE: {matrix}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

Produce 8-20 segments (15-35s each). Types:

  "pexels_broll"  — concept B-roll SPECIFIC to this topic's visual world
  "wiki_image"    — Wikipedia image for a concept, person, or tool mentioned
  "text_card"     — definition, key term, or important takeaway
  "code_card"     — code snippet (only when script references actual code)
  "stat_card"     — big-number fact
  "chapter_card"  — section transition (max 3 total)

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_broll", "duration_seconds": 20,
  "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "", "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "", "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules:
- pexels_query: 4-6 specific words tied to this exact concept.
- Mix: at least 30% pexels_broll, 2 text/stat cards, 1 chapter_card.
- Sum ≈ word_count / 130 * 60 seconds.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SHOT LIST GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_shot_list(api_key: str, has_trailer: bool, is_shorts: bool) -> list:
    mode           = st.session_state.get("mode_param", MODE_FILM)
    topic          = st.session_state.get("topic_param", "")
    matrix         = st.session_state.get("matrix_param", "")
    source_type    = st.session_state.get("source_param", "Original")
    script_text    = st.session_state.get("final_script_text", "")
    package        = st.session_state.get("package", {})
    script_outline = package.get("script_outline", [])

    if mode == MODE_FILM:
        char_names = [c["name"] for c in package.get("character_matrix", [])]
        prompt = _shot_prompt_film(topic, source_type, matrix, char_names,
                                   script_outline, script_text,
                                   has_trailer, is_shorts)
    elif mode == MODE_TECH:
        prompt = _shot_prompt_tech(topic, matrix, script_outline,
                                   script_text, is_shorts)
    else:
        prompt = _shot_prompt_edu(topic, matrix, script_outline,
                                  script_text, is_shorts)

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
# CLIP RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_clip(shot, mode,
                  used_urls: set, topic,
                  trailer_path: str | None, trailer_segs: list,
                  trailer_idx: list,
                  cw: int, ch: int,
                  is_shorts: bool):
    stype       = shot.get("type", "text_card")
    arr         = None
    clip_credit = ""

    def _fetch_unique(url):
        if not url or url in used_urls:
            return None
        a = _fetch_array(url, cw, ch)
        if a is not None:
            used_urls.add(url)
        return a

    # ── TRAILER CLIP ─────────────────────────────────────────────────────────
    if stype == "trailer_clip":
        stype = "tmdb_backdrop" # Safely forces fallback if called

    # ── WIKI IMAGE ───────────────────────────────────────────────────────────
    if stype in ("wiki_image", "tmdb_poster", "tmdb_backdrop", "tmdb_cast"):
        if stype == "tmdb_poster":
            queries = [topic, f"{topic} film", f"{topic} poster"]
        elif stype == "tmdb_backdrop":
            queries = [f"{topic} film", topic, f"{topic} movie scene"]
        elif stype == "tmdb_cast":
            name    = shot.get("cast_name", "")
            queries = [name, f"{name} actor"] if name else [topic]
        else:
            queries = [shot.get("wiki_title", topic), topic]

        url = _wiki_image_multi(queries)
        arr = _fetch_unique(url)
        if arr is not None:
            clip_credit = "Images courtesy: Wikipedia / Wikimedia Commons"

    # ── OPENVERSE B-ROLL ─────────────────────────────────────────────────────
    elif stype in ("pexels_broll", "openverse_image"):
        query = (shot.get("pexels_query")
                 or shot.get("image_query")
                 or shot.get("wiki_title")
                 or topic)
        for page in range(1, 4):
            url = _openverse_image(query, page=page)
            arr = _fetch_unique(url)
            if arr is not None:
                clip_credit = "Images courtesy: OpenVerse (CC licensed)"
                break

    # ── GENERATED CARDS ──────────────────────────────────────────────────────
    elif stype == "text_card":
        arr = make_text_card(
            shot.get("text_line1", shot.get("note", topic)),
            shot.get("text_line2", ""), mode=mode, cw=cw, ch=ch)

    elif stype == "stat_card":
        arr = make_stat_card(
            shot.get("stat_value", ""), shot.get("stat_desc", ""),
            mode=mode, cw=cw, ch=ch)

    elif stype == "code_card":
        arr = make_code_card(
            shot.get("code_snippet", "# No code provided"),
            shot.get("code_language", ""), cw=cw, ch=ch)

    elif stype == "chapter_card":
        arr = make_chapter_card(
            int(shot.get("chapter_num", 1) or 1),
            shot.get("chapter_title", shot.get("note", "")),
            mode=mode, cw=cw, ch=ch)

    # ── FALLBACK CHAIN ───────────────────────────────────────────────────────
    if arr is None:
        url = _openverse_image(topic + " cinematic")
        arr = _fetch_unique(url)
        if arr is not None:
            clip_credit = "Images courtesy: OpenVerse (CC licensed)"
        if arr is None:
            url = _wiki_image(topic)
            arr = _fetch_unique(url)
            if arr is not None:
                clip_credit = "Images courtesy: Wikipedia / Wikimedia Commons"
        if arr is None:
            arr = make_text_card(shot.get("note", topic), mode=mode, cw=cw, ch=ch)
            clip_credit = ""

    return arr, clip_credit


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ASSEMBLY WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _assembly_worker(audio_path, shot_list, mode, topic,
                     output_path, is_shorts, credit_override,
                     trailer_path=None, trailer_segs=None):
    try:
        cw, ch = _canvas(is_shorts)
        audio = AudioFileClip(audio_path)

        # ── FILM MODE: Direct Trailer Assembly ───────────────────────────
        if mode == MODE_FILM and trailer_path and os.path.exists(trailer_path):
            _write_progress(20, "Formatting custom trailer to fit screen...")
            video = VideoFileClip(trailer_path)

            if video.duration < audio.duration:
                from moviepy.video.fx.all import loop
                video = video.fx(loop, duration=audio.duration)
            else:
                start_time = 2.0 if video.duration > (audio.duration + 2.0) else 0.0
                video = video.subclip(start_time, start_time + audio.duration)

            scale = max(cw / video.w, ch / video.h)
            video = (video.resize(scale)
                          .crop(x_center=video.w * scale / 2,
                                y_center=video.h * scale / 2,
                                width=cw, height=ch))

            _assembled = video.set_audio(audio)
            
            _write_progress(70, "Adding credit overlay...")
            credit_text = credit_override or f"Courtesy: {topic}"
            
            def _overlay_frame(t):
                f = _assembled.get_frame(t)
                return _add_credit_overlay(f, credit_text, is_shorts, cw, ch)

            final = VideoClip(_overlay_frame, duration=_assembled.duration).set_audio(audio)

            _write_progress(85, "Rendering final MP4 (this takes ~60s)...")
            final.write_videofile(
                output_path, fps=FPS, codec="libx264",
                audio_codec="aac", preset="ultrafast",
                threads=2, logger=None,
            )
            _write_progress(100, "Done!", done=True)
            
            video.close()
            audio.close()
            final.close()
            return

        # ─────────────────────────────────────────────────────────────────────────────
        # ORIGINAL LOGIC: Kept intact for Tech/Edu modes (or if no trailer URL provided)
        # ─────────────────────────────────────────────────────────────────────────────
        
        trailer_segs = trailer_segs or []
        total        = len(shot_list)
        used_urls    = set()
        trailer_idx  = [0]
        clips        = []

        clip_credits  = []
        clip_starts   = []
        running_time  = 0.0

        for i, shot in enumerate(shot_list):
            pct = 20 + int(60 * i / total)
            _write_progress(pct,
                f"Building clip {i+1}/{total} [{shot.get('type','?')}] — "
                f"{shot.get('note','')}")

            duration = max(float(shot.get("duration_seconds", 12)), 5.0)
            zoom_in  = (i % 2 == 0)

            visual, auto_credit = _resolve_clip(
                shot, mode,
                used_urls, topic, trailer_path, trailer_segs,
                trailer_idx, cw, ch, is_shorts)

            if hasattr(visual, "duration"):
                clip = visual.set_duration(min(visual.duration, duration))
            else:
                clip = _zoom_clip(visual, duration, zoom_in)

            clip_starts.append(running_time)
            clip_credits.append(auto_credit)
            running_time += duration
            clips.append(clip)

        # ── Stitch ────────────────────────────────────────────────────────────
        _write_progress(82, "Stitching clips...")
        video = concatenate_videoclips(clips, method="compose")

        # ── Audio ─────────────────────────────────────────────────────────────
        _write_progress(87, "Attaching voiceover...")
        if video.duration < audio.duration:
            extra  = audio.duration - video.duration + clips[-1].duration
            last_frame = clips[-1].get_frame(0)
            filler = _zoom_clip(last_frame, extra, zoom_in=False)
            clips[-1] = filler
            video = concatenate_videoclips(clips, method="compose")

        _assembled = video.subclip(0, audio.duration).set_audio(audio)

        # ── Credit overlay ────────────────────────────────────────────────────
        _write_progress(91, "Adding credit overlay...")

        def _credit_at(t):
            if credit_override:
                return credit_override
            idx = bisect.bisect_right(clip_starts, t) - 1
            idx = max(0, min(idx, len(clip_credits) - 1))
            return clip_credits[idx]

        def _overlay_frame(t):
            f = _assembled.get_frame(t)
            return _add_credit_overlay(f, _credit_at(t), is_shorts, cw, ch)

        final = VideoClip(_overlay_frame, duration=_assembled.duration).set_audio(
            _assembled.audio)

        # ── Render ────────────────────────────────────────────────────────────
        _write_progress(93, "Rendering MP4 (this takes ~60s)...")
        final.write_videofile(
            output_path, fps=FPS, codec="libx264",
            audio_codec="aac", preset="ultrafast",
            threads=2, logger=None,
        )
        _write_progress(100, "Done!", done=True)
        
        video.close()
        audio.close()
        final.close()

    except Exception as e:
        _write_progress(0, "", done=True, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎬 SudoVid")
st.caption("AI-Powered Script, Voice & Video Engine")

# ─────────────────────────────────────────────────────────────────────────────
# KEY RESOLUTION  — secrets take priority; UI inputs are the fallback
# ─────────────────────────────────────────────────────────────────────────────
def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return ""

_gemini_from_secret = bool(_secret("GEMINI_API_KEY"))

with st.sidebar:
    st.header("🔑 API Key")

    if _gemini_from_secret:
        api_key = _secret("GEMINI_API_KEY")
        st.success("✓ Gemini (from Secrets)")
    else:
        api_key = st.text_input(
            "Gemini API Key", type="password",
            help="Add GEMINI_API_KEY to Streamlit Secrets to hide this field")
        if api_key:
            st.success("✓ Gemini API Key set")
        else:
            st.warning("⚠️ Gemini API Key required")

    st.divider()
    if st.button("Reset All Steps"):
        tmp_dir = tempfile.gettempdir()
        for f in os.listdir(tmp_dir):
            if f.startswith("trailer_"):
                try: os.remove(os.path.join(tmp_dir, f))
                except: pass
                
        st.session_state.clear()
        st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Parameters", "2. Ground Research", "3. Generated Script",
    "4. Voiceover", "5. Content Bundle", "6. Video Assembly",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Step 1: Set Project Parameters & Angle")
    st.info("Define the scope, tone, and your unique perspective before the AI conducts its research.")

    topic = st.text_input("Topic or Title",
                           placeholder="e.g., Project Hail Mary, Crowdstrike Outage")

    col_a, col_b = st.columns(2)
    with col_a:
        active_mode = st.selectbox("Content Mode", [MODE_FILM, MODE_TECH, MODE_EDU])
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
    st.caption("Upload your notes or type your perspective. The AI uses this to guide research.")
    angle_file = st.file_uploader("Upload rough draft (.txt)", type=["txt"])
    angle_text = st.text_area("Or type your angle here:", height=150,
                               placeholder="E.g., I think the main character's arc was ruined because…")

    if st.button("💾 Save Parameters & Proceed"):
        final_angle = angle_text.strip() or (
            angle_file.getvalue().decode("utf-8") if angle_file else "")
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
            st.success("✅ Parameters saved! Click **'2. Ground Research'**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GROUND RESEARCH
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Step 2: Targeted Intelligence Gathering")
    if "topic_param" not in st.session_state:
        st.info("Complete Step 1 first.")
    else:
        st.info(f"🌐 Searching the web to support your angle on "
                f"**{st.session_state['topic_param']}**.")
        if st.button("🔍 Execute Targeted Background Research"):
            if not api_key:
                st.warning("Gemini API Key required in sidebar.")
            else:
                with st.spinner("🌐 Searching…"):
                    st.session_state["research"] = perform_grounded_research(
                        topic       = st.session_state["topic_param"],
                        mode        = st.session_state["mode_param"],
                        source_type = st.session_state["source_param"],
                        angle       = st.session_state["angle_param"],
                        length      = st.session_state["length_param"],
                        api_key     = api_key,
                    )
        if "research" in st.session_state:
            st.success("✅ Research Complete")
            with st.expander("View Factual Briefing", expanded=False):
                st.markdown(st.session_state["research"])
            st.success("🎉 **Step 2 Complete!** Click **'3. Generated Script'**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GENERATED SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Step 3: Script Generation & Editing")
    if "research" not in st.session_state:
        st.info("Complete Step 2 first.")
    else:
        if st.button("🚀 Architect Refined Script"):
            with st.spinner(f"Synthesising for {st.session_state['length_param']}…"):
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
                with st.expander("Raw Output"):
                    st.text(p.get("raw"))
            else:
                st.success(f"### {p.get('viral_title')}")
                with st.expander("📊 Script Architecture Details", expanded=False):
                    st.markdown("#### 🌍 Thematic Resonance")
                    st.warning(f"**Analogous Event:** {p.get('thematic_resonance',{}).get('real_world_event')}")
                    st.write(p.get("thematic_resonance", {}).get("explanation"))
                    if st.session_state["mode_param"] == MODE_FILM:
                        for char in p.get("character_matrix", []):
                            st.markdown(
                                f"**{char['name']}** "
                                f"<span class='metric-badge'>{char['arc_score']}/10</span>",
                                unsafe_allow_html=True)

                st.markdown("### 📝 Conversational Script Editor")
                st.info("💡 Edit below as you want it spoken. Use commas or --- for natural pauses.")
                fs = p.get("full_script", {})
                default_text = "\n\n".join(filter(None, [
                    p.get("hook_script", ""), fs.get("intro", ""),
                    fs.get("act1", ""),       fs.get("act2", ""),
                    fs.get("act3", ""),       fs.get("outro", ""),
                ]))
                st.session_state["final_script_text"] = st.text_area(
                    "Final Polish:", value=default_text.strip(), height=400)
                st.download_button(
                    "📥 Download Text Script",
                    data=st.session_state["final_script_text"],
                    file_name=f"{p.get('viral_title','script').replace(' ','_').lower()}.txt",
                    mime="text/plain")
                st.success("🎉 **Step 3 Complete!** Click **'4. Voiceover'**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — VOICEOVER
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Step 4: AI Voiceover Studio")
    st.info("Turn your finalized script into professional audio.")

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
            st.warning("⚠️ No generated script found. Complete Steps 1-3 first.")
    else:
        uploaded_file = st.file_uploader("Upload .txt for Voiceover",
                                          type=["txt"], key="voice_upload")
        if uploaded_file:
            text_to_synthesize = uploaded_file.getvalue().decode("utf-8")
            st.success("File uploaded!")

    st.markdown("---")
    st.markdown("### Preview Text for Audio Generation")
    st.session_state["tab4_audio_text"] = st.text_area(
        "This exact text will be sent to the AI Voice:",
        value=text_to_synthesize, height=250)

    if st.button("🔊 Generate Voiceover"):
        if not st.session_state["tab4_audio_text"].strip():
            st.error("Text box is empty.")
        else:
            with st.spinner(f"Synthesising with {voice_option[1]}…"):
                audio_path = generate_audio_sync(
                    st.session_state["tab4_audio_text"], voice_option[0])
                if audio_path:
                    st.session_state["last_audio_path"] = audio_path
                    st.success("✅ Audio generated!")
                    st.audio(audio_path, format="audio/mp3")
                    with open(audio_path, "rb") as f:
                        st.download_button(
                            "📥 Download Audio (.mp3)", data=f,
                            file_name=f"{st.session_state.get('topic_param','voiceover').replace(' ','_').lower()}_voiceover.mp3",
                            mime="audio/mp3")
                else:
                    st.error("Audio generation failed. Check internet connection.")
            st.success("🎉 **Step 4 Complete!** Click **'5. Content Bundle'**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CONTENT BUNDLE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Step 5: YouTube Content Bundle")
    st.info("Generate SEO title, description, tags, hashtags, and thumbnail prompt.")

    bundle_source = st.radio("Script source for bundle:",
                              ["Use 'Generated Script' (from Tab 3)",
                               "Use 'Final Audio Text' (from Tab 4)"])
    if st.button("📦 Generate Content Bundle"):
        target_text = (st.session_state.get("final_script_text", "")
                       if bundle_source == "Use 'Generated Script' (from Tab 3)"
                       else st.session_state.get("tab4_audio_text", ""))
        if not api_key:
            st.error("⚠️ Gemini API Key required.")
        elif not target_text.strip():
            st.error("⚠️ Target text is empty.")
        else:
            with st.spinner("Generating YouTube metadata…"):
                st.session_state["yt_bundle"] = generate_youtube_bundle(api_key, target_text)

    if "yt_bundle" in st.session_state:
        bundle = st.session_state["yt_bundle"]
        if "error" in bundle:
            st.error(bundle["error"])
        else:
            st.success("✅ YouTube Bundle Generated!")
            st.markdown("### 📝 YouTube Metadata")
            st.text_input("**Viral Title**",  value=bundle.get("viral_title",  ""))
            st.text_area( "**Description**",   value=bundle.get("description",  ""), height=200)
            c1, c2 = st.columns(2)
            with c1:
                st.text_area("**Tags**", value=", ".join(bundle.get("tags", [])), height=100)
            with c2:
                st.text_area("**Hashtags**", value=" ".join(bundle.get("hashtags", [])), height=100)
            st.markdown("---")
            st.markdown("### 🎨 AI Thumbnail Prompt")
            st.caption("Paste into Midjourney, DALL-E, or Canva.")
            st.text_area("Image Prompt:", value=bundle.get("thumbnail_prompt", ""), height=100)
            st.success("🎉 **Step 5 Complete!** Click **'6. Video Assembly'**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — VIDEO ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Step 6: Video Assembly")

    mode       = st.session_state.get("mode_param", "")
    topic_val  = st.session_state.get("topic_param", "")
    length_val = st.session_state.get("length_param", "")
    is_shorts  = "Short" in length_val

    mode_labels = {MODE_FILM: "🎬 Film & Series",
                   MODE_TECH: "🔍 Tech / Investigative",
                   MODE_EDU:  "📚 Educational"}
    if mode:
        shorts_badge = " — 📱 **Shorts 9:16**" if is_shorts else " — 🖥️ **Landscape 16:9**"
        st.info(f"Mode: **{mode_labels.get(mode, mode)}** — Topic: **{topic_val}**{shorts_badge}")

    # Credit override
    override_on = st.checkbox(
        "✏️ Override automatic credits",
        value=False,
        help="By default credits are set automatically per clip source. Enable this to write your own single credit line for the whole video.")
    credit_override = ""
    if override_on:
        credit_override = st.text_input(
            "Custom credit text",
            value=f"Courtesy: {topic_val}" if topic_val else "Courtesy: Studio Name",
            help="Shorts: top centre white pill. Long format: bottom left small white text.")
    else:
        st.caption("🤖 Credits set automatically per clip.")

    st.markdown("---")

    # Pre-flight checks
    audio_path_v  = st.session_state.get("last_audio_path", "")
    script_text_v = st.session_state.get("final_script_text", "")

    missing = []
    if not audio_path_v or not os.path.exists(audio_path_v):
        missing.append("✗ No audio — complete Step 4 first")
    if not script_text_v:
        missing.append("✗ No script — complete Step 3 first")
    if not topic_val:
        missing.append("✗ No topic — complete Step 1 first")
    if not api_key:
        missing.append("✗ Gemini API Key required (sidebar)")

    if missing:
        for m in missing:
            st.warning(m)
    else:
        if mode == MODE_FILM:
            st.caption("ℹ️ Paste a direct video link below, or leave it blank to have the AI search for the official trailer.")
            
            custom_trailer_url = st.text_input(
                "🔗 Exact Trailer URL",
                placeholder="Leave blank to auto-search, or paste a YouTube/IMDb link...",
            )
            
            if st.button("🎞️ Fetch Trailer"):
                with st.spinner("Preparing trailer…"):
                    safe_name = re.sub(r"[^a-z0-9]", "_", topic_val.lower())
                    tp = os.path.join(tempfile.gettempdir(), f"trailer_{safe_name}_custom.mp4")
                    
                    target_url = custom_trailer_url.strip()
                    
                    # THE BRILLIANT FIX: Use Gemini to find the URL if the user didn't provide one
                    if not target_url:
                        st.info("🤖 Asking AI to find the official theatrical trailer...")
                        
                        search_prompt = f"""
                        Search the web for the OFFICIAL cinematic theatrical movie trailer for "{topic_val}".
                        You must find the exact YouTube or IMDb video URL.
                        CRITICAL: Exclude all video games, VR experiences, interviews, fan reactions, and IGN FanFests.
                        Return ONLY the raw URL as plain text. Do not add any conversational text or markdown formatting.
                        """
                        
                        target_url = call_gemini(
                            api_key=api_key, 
                            prompt=search_prompt, 
                            system_instruction="You are a strict URL-finding bot. Output ONLY a valid http/https URL.", 
                            use_search=True
                        )
                        
                        if target_url and target_url.strip().startswith("http"):
                            st.success(f"🔍 AI found trailer: {target_url.strip()}")
                        else:
                            st.error("❌ AI could not confidently find a trailer URL. Please paste one manually.")
                            target_url = None

                    # Proceed to download the URL (whether provided manually or found by AI)
                    if target_url:
                        if _download_custom_video(target_url.strip(), tp):
                            st.session_state["trailer_path"] = tp
                            st.success("✅ Trailer downloaded successfully! You can now Start Video Assembly.")
                        else:
                            st.session_state["trailer_path"] = None
                            st.error("❌ Failed to download the video. YouTube may have blocked the connection. Try pasting an IMDb link.")

        # Shot list
        if st.button("🎬 Generate Shot List"):
            trailer_available = bool(st.session_state.get("trailer_path"))
            with st.spinner("Gemini is planning your shot list…"):
                st.session_state["shot_list"] = generate_shot_list(
                    api_key, has_trailer=trailer_available, is_shorts=is_shorts)

        if "shot_list" in st.session_state:
            sl        = st.session_state["shot_list"]
            total_dur = sum(s.get("duration_seconds", 12) for s in sl)
            st.success(f"Shot list ready — **{len(sl)} clips**, ~**{int(total_dur)}s** total")

            type_icons = {
                "trailer_clip":    "🎬",
                "wiki_image":      "📖",
                "tmdb_poster":     "🖼️",
                "tmdb_backdrop":   "🎞️",
                "tmdb_cast":       "👤",
                "pexels_broll":    "🌆",
                "openverse_image": "🌆",
                "text_card":       "📝",
                "stat_card":       "📊",
                "code_card":       "💻",
                "chapter_card":    "📌",
            }
            with st.expander("📋 View Shot List", expanded=False):
                for s in sl:
                    icon   = type_icons.get(s.get("type", ""), "▪️")
                    detail = (s.get("image_query") or s.get("pexels_query") or
                              s.get("cast_name")   or s.get("wiki_title")   or
                              s.get("text_line1")  or s.get("stat_value")   or
                              s.get("chapter_title") or s.get("note", ""))
                    st.markdown(
                        f"`{s.get('segment_index',0):02d}` {icon} "
                        f"**{s.get('type','')}** — "
                        f"{s.get('duration_seconds','?')}s — _{detail}_")

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
                            audio_path      = audio_path_v,
                            shot_list       = sl,
                            mode            = mode,
                            topic           = topic_val,
                            output_path     = output_path,
                            is_shorts       = is_shorts,
                            credit_override = credit_override,
                            trailer_path    = st.session_state.get("trailer_path"),
                            trailer_segs    = [], # Segments removed; gracefully bypass
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
                                "📥 Download MP4", data=f,
                                file_name=f"{topic_val.replace(' ','_').lower()}"
                                          f"{'_shorts' if is_shorts else ''}_video.mp4",
                                mime="video/mp4")
                    if st.button("🔄 Reset & Assemble Again"):
                        for k in ("shot_list", "assembly_running",
                                  "assembly_output", "trailer_path", "trailer_segs"):
                            st.session_state.pop(k, None)
                        st.rerun()
                else:
                    pct = prog.get("pct", 0)
                    msg = prog.get("msg", "Working…")
                    st.progress(pct / 100, text=f"{msg} ({pct}%)")
                    st.caption("⏳ Keep this tab open.")
                    time.sleep(3)
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("SudoVid v2.0 | AI-Powered Script, Voice & Video Engine | Zero API Keys Required")