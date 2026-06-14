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
from moviepy.editor import (
    AudioFileClip, concatenate_videoclips, VideoFileClip, ImageClip
)
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
    .media-card {
        background-color: #f0f9ff; padding: 12px; border-radius: 8px;
        border: 1px solid #bae6fd; margin-bottom: 8px; font-size: 0.85em;
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

CANVAS_LANDSCAPE = (1280, 720)
CANVAS_SHORTS    = (1080, 1920)

FPS             = 24
WIKI_THUMB      = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
OPENVERSE_BASE  = "https://api.openverse.org/v1/images/"
PEXELS_IMAGE_BASE = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_BASE = "https://api.pexels.com/videos/search"
PROGRESS_FILE   = os.path.join(tempfile.gettempdir(), "sa_video_progress.json")

# Temp directory structure
TMP_ROOT         = tempfile.gettempdir()
UPLOAD_DIR       = os.path.join(TMP_ROOT, "sudovid_uploads")
KEYFRAME_DIR     = os.path.join(TMP_ROOT, "sudovid_keyframes")
PEXELS_CACHE_DIR = os.path.join(TMP_ROOT, "sudovid_pexels_cache")

for _d in (UPLOAD_DIR, KEYFRAME_DIR, PEXELS_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# Model name — Gemini 3.5 Flash (stable, current flagship as of June 2026)
GEMINI_MODEL = "gemini-3.5-flash"

_DARK_BG  = (10, 12, 20)
_ACCENT   = (37,  99, 235)
_TECH_ACC = (234, 88,  12)
_EDU_ACC  = (22, 163,  74)
_TEXT_HI  = (240, 245, 255)
_TEXT_LO  = (148, 163, 184)

# File limits
VIDEO_MAX_BYTES = 500 * 1024 * 1024   # 500 MB
IMAGE_MAX_BYTES =  50 * 1024 * 1024   #  50 MB
ALLOWED_VIDEO_TYPES = ["mp4", "mov", "webm", "mkv"]
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


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
# GEMINI WRAPPER  — all calls use gemini-3.5-flash
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(api_key, prompt, system_instruction="", use_search=False,
                is_json=False, image_parts=None):
    """
    Unified Gemini call. Supports:
      - Text-only calls (default)
      - Vision calls: pass image_parts as list of genai.types.Part objects
      - JSON mode: is_json=True sets response_mime_type
      - Grounded search: use_search=True (REST API, cannot combine with vision)
    """
    if not use_search:
        genai.configure(api_key=api_key)
        gen_config = {"response_mime_type": "application/json"} if is_json else None
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_instruction or None,
            generation_config=gen_config,
        )
        contents = []
        if image_parts:
            contents.extend(image_parts)
        contents.append(prompt)

        for delay in [1, 2, 4, 8, 16]:
            try:
                return model.generate_content(contents).text
            except Exception as e:
                if delay == 16:
                    return f"Error: {str(e)}"
                time.sleep(delay)
    else:
        # Grounded search via REST (cannot send images here)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={api_key}"
        )
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction or ""}]},
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
# MEDIA UPLOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(uploaded_file, media_type: str) -> dict | None:
    """
    Validate and persist an uploaded file to UPLOAD_DIR.
    Returns metadata dict or None on failure.
    media_type: 'video' | 'image'
    """
    if uploaded_file is None:
        return None

    size = uploaded_file.size
    limit = VIDEO_MAX_BYTES if media_type == "video" else IMAGE_MAX_BYTES
    limit_label = "500 MB" if media_type == "video" else "50 MB"

    if size > limit:
        st.warning(
            f"⚠️ '{uploaded_file.name}' exceeds the {limit_label} limit "
            f"({size / 1024 / 1024:.1f} MB) and was skipped."
        )
        return None

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    allowed = ALLOWED_VIDEO_TYPES if media_type == "video" else ALLOWED_IMAGE_TYPES
    if ext not in allowed:
        st.warning(f"⚠️ '{uploaded_file.name}' has unsupported extension '.{ext}' — skipped.")
        return None

    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", uploaded_file.name)
    dest = os.path.join(UPLOAD_DIR, safe_name)

    # Avoid re-writing if identical file already saved
    if not os.path.exists(dest) or os.path.getsize(dest) != size:
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())

    meta = {
        "filename":   safe_name,
        "path":       dest,
        "size_mb":    round(size / 1024 / 1024, 2),
        "ext":        ext,
        "media_type": media_type,
    }
    return meta


def _extract_video_keyframes(video_path: str, interval_sec: float = 10.0) -> list[str]:
    """
    Extract one keyframe every `interval_sec` from a video.
    Returns list of saved JPEG paths in KEYFRAME_DIR.
    Caps at 20 frames to stay within Gemini context limits.
    """
    frames = []
    try:
        vc = VideoFileClip(video_path)
        duration = vc.duration
        times = list(np.arange(0, duration, interval_sec))[:20]
        base = os.path.splitext(os.path.basename(video_path))[0]
        for i, t in enumerate(times):
            frame_arr = vc.get_frame(t)
            img = Image.fromarray(frame_arr.astype(np.uint8))
            out = os.path.join(KEYFRAME_DIR, f"{base}_kf{i:03d}.jpg")
            img.save(out, "JPEG", quality=75)
            frames.append(out)
        vc.close()
    except Exception:
        pass
    return frames


def _build_vision_parts(file_meta: dict) -> list:
    """
    Build Gemini vision Parts for a single media file.
    Videos: send extracted keyframes as inline images.
    Images: send directly as inline image.
    Returns list of genai.types.Part objects.
    """
    parts = []
    try:
        if file_meta["media_type"] == "image":
            with open(file_meta["path"], "rb") as f:
                data = f.read()
            mime = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png",  "webp": "image/webp",
            }.get(file_meta["ext"], "image/jpeg")
            parts.append({"inline_data": {"mime_type": mime, "data": data}})

        else:  # video — send keyframes
            kf_paths = _extract_video_keyframes(file_meta["path"], interval_sec=10.0)
            for kf in kf_paths:
                with open(kf, "rb") as f:
                    data = f.read()
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": data}})

    except Exception:
        pass
    return parts


def analyse_uploaded_media(api_key: str, media_list: list[dict]) -> list[dict]:
    """
    Send all uploaded files to Gemini Vision for analysis.
    Returns a list of dicts enriched with AI analysis fields.
    Each dict: { filename, media_type, duration_seconds (video only),
                 dominant_subjects, mood, color_palette,
                 suggested_use, content_tags }
    """
    if not media_list:
        return []

    genai.configure(api_key=api_key)

    # Build combined prompt + all vision parts in ONE call to save quota
    all_parts = []
    file_labels = []
    for i, meta in enumerate(media_list):
        label = f"FILE_{i+1}: {meta['filename']} ({meta['media_type']}, {meta['size_mb']} MB)"
        file_labels.append(label)
        vparts = _build_vision_parts(meta)
        # Prepend a text marker so Gemini knows which file the frames belong to
        all_parts.append({"text": label})
        all_parts.extend(vparts)

    files_block = "\n".join(file_labels)
    prompt_text = f"""
You are a professional video editor analysing uploaded media files for a YouTube video.

Files provided:
{files_block}

For EACH file, provide a structured JSON analysis.

Return ONLY a valid JSON array — one object per file in the same order.

Schema per object:
{{
  "filename": "exact filename from the list",
  "media_type": "video" or "image",
  "duration_seconds": 0,
  "dominant_subjects": "1-2 sentence description of main visual subjects",
  "mood": "one of: cinematic, energetic, calm, dramatic, technical, educational, abstract",
  "color_palette": "brief description e.g. warm golden tones, cool blues",
  "suggested_use": "specific recommended use in a YouTube video (hook, b-roll, transition, etc.)",
  "content_tags": ["tag1", "tag2", "tag3"]
}}

For images, set duration_seconds to 0.
For videos, estimate duration from keyframe count (keyframe_interval = 10s).
Be specific and actionable — these descriptions feed directly into shot list planning.
"""
    all_parts.append({"text": prompt_text})

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )

    for delay in [1, 2, 4, 8, 16]:
        try:
            # Use the low-level Part approach compatible with google-generativeai SDK
            from google.generativeai.types import content_types
            response = model.generate_content(all_parts)
            raw = response.text.replace("```json", "").replace("```", "").strip()
            analyses = json.loads(raw)
            # Merge back into media_list metadata
            result = []
            for i, meta in enumerate(media_list):
                merged = dict(meta)
                if i < len(analyses):
                    merged.update(analyses[i])
                result.append(merged)
            return result
        except Exception as e:
            if delay == 16:
                # Return metadata without AI analysis rather than failing hard
                return [dict(m, dominant_subjects="", mood="", color_palette="",
                             suggested_use="", content_tags=[])
                        for m in media_list]
            time.sleep(delay)

    return media_list


# ─────────────────────────────────────────────────────────────────────────────
# PEXELS API  — Images + Videos (unified key)
# ─────────────────────────────────────────────────────────────────────────────

def _pexels_image_url(query: str, pexels_key: str, orientation="landscape") -> str | None:
    """Fetch highest-quality Pexels image URL for query."""
    if not pexels_key:
        return None
    try:
        r = requests.get(
            PEXELS_IMAGE_BASE,
            headers={"Authorization": pexels_key},
            params={"query": query, "per_page": 5, "orientation": orientation},
            timeout=12,
        ).json()
        photos = r.get("photos", [])
        if not photos:
            return None
        # Prefer 'original', fall back to 'large2x', then 'large'
        p = photos[0]["src"]
        return p.get("original") or p.get("large2x") or p.get("large")
    except Exception:
        return None


def _pexels_video_download(query: str, pexels_key: str,
                            out_dir: str, orientation="landscape") -> str | None:
    """
    Search Pexels Videos API, download highest-quality MP4 to out_dir.
    Returns local path or None.
    Caches by query slug to avoid re-downloading.
    """
    if not pexels_key:
        return None

    safe_q = re.sub(r"[^a-z0-9]", "_", query.lower())[:60]
    cached = os.path.join(out_dir, f"pexels_{safe_q}.mp4")
    if os.path.exists(cached) and os.path.getsize(cached) > 100_000:
        return cached

    try:
        r = requests.get(
            PEXELS_VIDEO_BASE,
            headers={"Authorization": pexels_key},
            params={"query": query, "per_page": 5, "orientation": orientation},
            timeout=15,
        ).json()
        videos = r.get("videos", [])
        if not videos:
            return None

        # Pick best quality: sort video_files by width descending
        video_files = videos[0].get("video_files", [])
        video_files_sorted = sorted(
            [vf for vf in video_files if vf.get("file_type") == "video/mp4"],
            key=lambda x: x.get("width", 0),
            reverse=True,
        )
        if not video_files_sorted:
            return None

        download_url = video_files_sorted[0]["link"]
        resp = requests.get(download_url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(cached, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 19):
                f.write(chunk)

        if os.path.getsize(cached) > 100_000:
            return cached
    except Exception:
        pass
    return None


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
                             source_type, length, api_key, media_analysis=None):
    personas = {
        MODE_FILM: "Master YouTube Film Critic. Focus on narrative, character arcs, and thematic depth.",
        MODE_TECH: "Investigative Tech YouTuber. Focus on clarity, impact, and engaging storytelling.",
        MODE_EDU:  "Senior Developer turned YouTuber. Explain things naturally, like a mentor talking to a junior.",
    }

    media_block = ""
    if media_analysis:
        lines = []
        for m in media_analysis:
            line = (
                f"  - {m.get('filename')} [{m.get('media_type')}]: "
                f"{m.get('dominant_subjects', '')} | "
                f"Mood: {m.get('mood', '')} | "
                f"Suggested use: {m.get('suggested_use', '')}"
            )
            lines.append(line)
        media_block = (
            "\nUSER UPLOADED MEDIA (incorporate these naturally into the script structure):\n"
            + "\n".join(lines)
        )

    prompt = f"""
    TOPIC: {topic}
    SOURCE TYPE: {source_type}
    VIDEO LENGTH: {length}
    CREATOR'S DRAFT / UNIQUE ANGLE: {angle}
    SELECTED MATRIX (Tone/Style): {matrix}
    TARGETED RESEARCH: {research}
    {media_block}

    TASK: You are a professional, conversational YouTube scriptwriter. Your goal is
    to refine the "CREATOR'S DRAFT" into a highly engaging, human-sounding script
    ready for voiceover.

    CRITICAL INSTRUCTIONS:
    1. LENGTH ADAPTATION: Target video length is '{length}'.
       - YouTube Short: under 150 words total.
       - Mid-length or Deep Dive: flesh out arguments with natural pacing.
    2. HUMAN TONE: Conversational, rhetorical questions, natural transitions. No robotic lists.
    3. ANGLE-FIRST: Preserve creator's unique perspective. Research supports, not replaces.
    4. If USER UPLOADED MEDIA is provided, reference those visual moments naturally
       in the script structure so editors know when to cut to them.
    5. ESCAPE all double quotes inside script text properly.

    JSON SCHEMA:
    {{
      "thematic_resonance": {{ "real_world_event": "String", "explanation": "String" }},
      "character_matrix": [ {{ "name": "Name", "role": "Main/Side", "arc_score": 0, "ghost_vs_truth": "String" }} ],
      "technical_report": {{ "script": 0, "direction": 0, "editing": 0, "acting": 0 }},
      "viral_title": "String",
      "hook_script": "String",
      "full_script": {{
          "intro": "String", "act1": "String", "act2": "String",
          "act3": "String", "outro": "String"
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
        return {"error": f"Synthesis failed. Error: {str(e)}", "raw": result}


def generate_youtube_bundle(api_key, script_text):
    prompt = f"""
    Analyse this YouTube script and create a complete SEO and packaging bundle.

    SCRIPT:
    {script_text}

    JSON SCHEMA:
    {{
        "viral_title": "String",
        "description": "String",
        "tags": ["tag1", "tag2"],
        "hashtags": ["#tag1", "#tag2"],
        "thumbnail_prompt": "String"
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
# IMAGE FIT
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
# EXTERNAL IMAGE SOURCES  (key-free fallbacks)
# ─────────────────────────────────────────────────────────────────────────────

def _openverse_image(query: str, page: int = 1) -> str | None:
    try:
        r = requests.get(
            OPENVERSE_BASE,
            params={"q": query, "license_type": "commercial",
                    "page_size": 5, "page": page},
            headers={"User-Agent": "SudoVid/2.0"},
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
# TRAILER DOWNLOAD & SEGMENTATION  (Film mode)
# ─────────────────────────────────────────────────────────────────────────────

def _hd_trailers_slug(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[':!?,\.]", "", slug)
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _parse_hd_trailers_page(html: str) -> str | None:
    try:
        from bs4 import BeautifulSoup
        import json as _json
        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = _json.loads(script.string or "")
                url  = (data.get("trailer") or {}).get("contentUrl", "")
                if url and "movietrailers.apple.com" in url:
                    return re.sub(r"h(1080|480)p", "h720p", url)
            except Exception:
                pass
        for quality in ("trailer-quality-720", "trailer-quality-1080", "trailer-quality-480"):
            for td in soup.find_all("td", class_=quality):
                a = td.find("a", href=re.compile(r"movietrailers\.apple\.com"))
                if a:
                    return a["href"]
        for a in soup.find_all("a", href=re.compile(r"movietrailers\.apple\.com.*h720p")):
            return a["href"]
    except Exception:
        pass
    return None


def _download_apple_cdn(apple_url: str, out_path: str) -> bool:
    try:
        hdrs = {"User-Agent": "QuickTime/7.7.3 (qtver=7.7.3;os=Windows NT 6.1)"}
        with requests.get(apple_url, headers=hdrs, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 19):
                    f.write(chunk)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 100_000
    except Exception:
        return False


def _apple_trailers_xml(topic: str) -> str | None:
    import difflib
    import xml.etree.ElementTree as ET
    XML_URL = "https://trailers.apple.com/trailers/home/xml/current.xml"
    cache = _apple_trailers_xml.__dict__.setdefault("_cache", {})
    xml_text = cache.get(XML_URL)
    if not xml_text:
        try:
            r = requests.get(XML_URL, headers={"User-Agent": "QuickTime/7.7.3"}, timeout=20)
            r.raise_for_status()
            xml_text = r.text
            cache[XML_URL] = xml_text
        except Exception:
            return None
    try:
        root = ET.fromstring(xml_text)
        topic_low = topic.lower()
        best_url = None
        best_score = 0.0
        for movie in root.findall(".//movieinfo"):
            title_el = movie.find("info/title")
            if title_el is None or not title_el.text:
                continue
            score = difflib.SequenceMatcher(None, topic_low, title_el.text.lower()).ratio()
            if score < 0.60:
                continue
            video_el = movie.find("video")
            if video_el is not None:
                for quality in ("hd720p", "hd1080p", "hd480p"):
                    src_el = video_el.find(f'.//videosize[@type="{quality}"]/src')
                    if src_el is not None and src_el.text and "movietrailers.apple.com" in src_el.text:
                        if score > best_score:
                            best_score = score
                            best_url = src_el.text
                        break
    except ET.ParseError:
        return None
    return best_url


def _imdb_trailer_url(topic: str) -> str | None:
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "simulate": True, "forceurl": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            search_results = ydl.extract_info(f"ytsearch:{topic} trailer imdb", download=False)
            if search_results and "entries" in search_results and search_results["entries"]:
                return search_results["entries"][0]["url"]
    except Exception:
        pass
    return None


def _download_imdb_trailer(imdb_video_url: str, out_path: str) -> bool:
    try:
        import yt_dlp
        opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
            "outtmpl": out_path, "quiet": True, "no_warnings": True,
            "merge_output_format": "mp4", "retries": 5, "socket_timeout": 30,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([imdb_video_url])
        return os.path.exists(out_path) and os.path.getsize(out_path) > 100_000
    except Exception:
        return False


def _internet_archive_trailer(topic: str) -> str | None:
    try:
        r = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": (f'title:("{topic}") AND '
                      '(subject:"trailer" OR subject:"movie trailer") AND mediatype:movies'),
                "fl[]": ["identifier", "title"], "rows": 5, "output": "json",
            },
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        for doc in docs:
            identifier = doc.get("identifier", "")
            if not identifier:
                continue
            meta = requests.get(f"https://archive.org/metadata/{identifier}", timeout=10).json()
            files = meta.get("files", [])
            for name_filter, exts in (
                (lambda n: "trailer" in n, ("mp4", "mov")),
                (lambda n: True,           ("mp4",)),
            ):
                for ext in exts:
                    for f in files:
                        fname = f.get("name", "")
                        if fname.lower().endswith(f".{ext}") and name_filter(fname.lower()):
                            return f"https://archive.org/download/{identifier}/{fname}"
    except Exception:
        pass
    return None


def _download_trailer(topic: str) -> str | None:
    safe_name = re.sub(r"[^a-z0-9]", "_", topic.lower())
    out_path  = os.path.join(TMP_ROOT, f"trailer_{safe_name}.mov")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100_000:
        return out_path
    try:
        slug = _hd_trailers_slug(topic)
        r = requests.get(
            f"https://www.hd-trailers.net/movie/{slug}/",
            headers={"User-Agent": "Mozilla/5.0 (compatible; SudoVid/2.0)"},
            timeout=15,
        )
        if r.status_code == 200:
            apple_url = _parse_hd_trailers_page(r.text)
            if apple_url and _download_apple_cdn(apple_url, out_path):
                return out_path
    except Exception:
        pass
    try:
        apple_url = _apple_trailers_xml(topic)
        if apple_url and _download_apple_cdn(apple_url, out_path):
            return out_path
    except Exception:
        pass
    try:
        imdb_url = _imdb_trailer_url(topic)
        if imdb_url:
            out_path_mp4 = out_path.replace(".mov", ".mp4")
            if _download_imdb_trailer(imdb_url, out_path_mp4):
                return out_path_mp4
    except Exception:
        pass
    try:
        archive_url = _internet_archive_trailer(topic)
        if archive_url:
            out_path_mp4 = out_path.replace(".mov", "_archive.mp4")
            if _download_apple_cdn(archive_url, out_path_mp4):
                return out_path_mp4
    except Exception:
        pass
    return None


def _extract_trailer_segments(video_path: str, num_segments: int = 6,
                               seg_duration: float = 8.0) -> list:
    try:
        vc = VideoFileClip(video_path)
        total = vc.duration
        vc.close()
        usable_start = 2.0
        usable_end   = max(usable_start + seg_duration, total - 5.0)
        usable_range = usable_end - usable_start - seg_duration
        if usable_range <= 0:
            return []
        step = usable_range / max(num_segments - 1, 1)
        segs = []
        for i in range(num_segments):
            start = usable_start + i * step
            end   = min(start + seg_duration, usable_end)
            if end - start >= 3.0:
                segs.append((round(start, 1), round(end, 1)))
        return segs
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# USER UPLOAD VIDEO — loop to fill duration
# ─────────────────────────────────────────────────────────────────────────────

def _loop_video_clip(video_path: str, target_duration: float, cw: int, ch: int):
    """
    Load a user-uploaded video, resize to canvas, loop it to fill target_duration.
    Returns a MoviePy VideoClip.
    """
    vc = VideoFileClip(video_path)

    # Resize to canvas
    if vc.w != cw or vc.h != ch:
        src_ratio = vc.w / vc.h
        tgt_ratio = cw / ch
        both_l = src_ratio >= 1.0 and tgt_ratio >= 1.0
        both_p = src_ratio <  1.0 and tgt_ratio <  1.0
        if both_l or both_p:
            scale = max(cw / vc.w, ch / vc.h)
            vc = vc.resize(scale).crop(x_center=vc.w * scale / 2,
                                       y_center=vc.h * scale / 2,
                                       width=cw, height=ch)
        else:
            scale = min(cw / vc.w, ch / vc.h)
            vc = vc.resize(scale)
            # Pad with blurred background
            frame0 = Image.fromarray(vc.get_frame(0).astype(np.uint8))
            bg_arr = _fit_to_canvas(frame0, cw, ch)
            bg_clip = ImageClip(bg_arr, duration=target_duration)
            from moviepy.editor import CompositeVideoClip
            ox = (cw - vc.w) // 2
            oy = (ch - vc.h) // 2
            vc = CompositeVideoClip(
                [bg_clip, vc.set_position((ox, oy))],
                size=(cw, ch)
            ).set_duration(min(vc.duration, target_duration))

    clip_dur = vc.duration
    if clip_dur <= 0:
        vc.close()
        raise ValueError("Video has zero duration")

    if clip_dur >= target_duration:
        return vc.subclip(0, target_duration)

    # Loop: tile the clip until we exceed target_duration
    loops_needed = math.ceil(target_duration / clip_dur)
    from moviepy.editor import concatenate_videoclips
    looped = concatenate_videoclips([vc] * loops_needed, method="compose")
    return looped.subclip(0, target_duration)


# ─────────────────────────────────────────────────────────────────────────────
# CREDIT OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def _add_credit_overlay(frame_arr: np.ndarray, credit_text: str,
                         is_shorts: bool, cw: int, ch: int) -> np.ndarray:
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
    scale = ch / 720
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
    scale = ch / 720
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
    scale = ch / 720
    draw.text((cw // 2, ch // 2 - int(60 * scale)), stat,
              font=_font(max(48, int(110 * scale)), bold=True),
              fill=_TEXT_HI, anchor="mm")
    draw.text((cw // 2, ch // 2 + int(70 * scale)), description,
              font=_font(max(18, int(36 * scale))), fill=_TEXT_LO, anchor="mm")
    return np.array(img)


def make_code_card(snippet, language="", cw=1280, ch=720):
    img  = Image.new("RGB", (cw, ch), (18, 20, 30))
    draw = ImageDraw.Draw(img)
    scale = ch / 720
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
        draw.text((int(50 * scale), y), str(i + 1), font=mf, fill=(64, 74, 94), anchor="rm")
        stripped = line.lstrip()
        if stripped.startswith(("#", "//")):
            col = (106, 153, 85)
        elif any(stripped.startswith(k) for k in (
            "def ", "class ", "import ", "from ", "return ",
            "const ", "let ", "var ", "function ", "async ")):
            col = (197, 134, 192)
        elif stripped.startswith(("'", '"', "`", "f'")):
            col = (206, 145, 120)
        else:
            col = _TEXT_HI
        draw.text((int(64 * scale), y), line, font=mf, fill=col)
        y += line_h
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# KEN BURNS ZOOM
# ─────────────────────────────────────────────────────────────────────────────

def _zoom_clip(arr: np.ndarray, duration: float, zoom_in: bool = True):
    pil = Image.fromarray(arr)
    h0, w0 = arr.shape[:2]

    def make_frame(t):
        p     = t / max(duration, 0.001)
        scale = 1.0 + 0.04 * (p if zoom_in else (1 - p))
        nw, nh = int(w0 * scale), int(h0 * scale)
        res   = np.array(pil.resize((nw, nh), Image.BILINEAR))
        x0_ = (nw - w0) // 2
        y0_ = (nh - h0) // 2
        return res[y0_:y0_ + h0, x0_:x0_ + w0]

    return VideoClip(make_frame, duration=duration)


# ─────────────────────────────────────────────────────────────────────────────
# SHOT LIST PROMPTS  — now media-aware
# ─────────────────────────────────────────────────────────────────────────────

def _media_analysis_block(media_analysis: list[dict]) -> str:
    """Format uploaded media analysis for injection into shot prompts."""
    if not media_analysis:
        return ""
    lines = ["USER UPLOADED MEDIA (PREFER these over external sources):"]
    for m in media_analysis:
        tags = ", ".join(m.get("content_tags", []))
        dur  = f" | {m.get('duration_seconds', 0):.0f}s" if m.get("media_type") == "video" else ""
        lines.append(
            f"  filename={m['filename']} type={m['media_type']}{dur} | "
            f"{m.get('dominant_subjects', '')} | mood={m.get('mood', '')} | "
            f"use={m.get('suggested_use', '')} | tags=[{tags}]"
        )
    lines.append(
        "\nINSTRUCTION: Use 'user_upload_video' or 'user_upload_image' shot types "
        "and set 'upload_filename' to the exact filename above for any shot where "
        "an uploaded file is a good visual match. Only fall back to pexels_video, "
        "pexels_image, or generated cards when no upload fits."
    )
    return "\n".join(lines)


def _shot_prompt_film(topic, source_type, matrix, character_names,
                      script_outline, script_text, has_trailer, is_shorts,
                      media_analysis):
    char_list  = ", ".join(character_names) if character_names else "none"
    outline    = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    media_blk  = _media_analysis_block(media_analysis)

    if is_shorts:
        trailer_note = (
            'TRAILER AVAILABLE: Yes — use "trailer_clip" for action/plot moments.'
            if has_trailer else
            'TRAILER AVAILABLE: No — use tmdb_backdrop and tmdb_poster as primary.'
        )
        return f"""
You are a video editor cutting a YouTube SHORT (vertical 9:16, under 60 seconds).

FILM TITLE: {topic}
{trailer_note}
KNOWN CAST: {char_list}
SCRIPT: {script_text}

{media_blk}

Produce EXACTLY 3-6 segments, each 5-12 seconds.

Visual types for Shorts:
  "user_upload_video"  — user-provided video clip (HIGHEST PRIORITY if filename matches)
  "user_upload_image"  — user-provided image (HIGHEST PRIORITY if filename matches)
  "trailer_clip"       — official trailer segment
  "tmdb_poster"        — official film poster
  "tmdb_backdrop"      — cinematic scene still
  "text_card"          — punchy one-liner (max 1 total)

Return ONLY a valid JSON array. Schema per item:
{{
  "segment_index": 0,
  "type": "user_upload_video",
  "upload_filename": "",
  "duration_seconds": 8,
  "trailer_timestamp": 0,
  "pexels_query": "",
  "cast_name": "",
  "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "",
  "wiki_title": "",
  "code_snippet": "", "code_language": "",
  "note": ""
}}

Rules:
- upload_filename: exact filename from USER UPLOADED MEDIA list, or "" if not using an upload.
- Sum of duration_seconds must be 30-58 seconds total.
- Prefer user uploads where mood/content matches the script moment.
"""
    else:
        trailer_note = (
            'TRAILER AVAILABLE: Yes — use "trailer_clip" for high-energy moments.'
            if has_trailer else
            'TRAILER AVAILABLE: No — skip trailer_clip entirely.'
        )
        return f"""
You are a video editor planning a YouTube FILM / SERIES REVIEW video.

FILM TITLE: {topic}
SOURCE TYPE: {source_type}
TONE MATRIX: {matrix}
{trailer_note}
KNOWN CAST: {char_list}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

{media_blk}

Produce 8-20 segments (15-35s each). Visual types:
  "user_upload_video"  — user-provided video (HIGHEST PRIORITY)
  "user_upload_image"  — user-provided image (HIGHEST PRIORITY)
  "trailer_clip"       — official trailer segment
  "tmdb_poster"        — film poster (opening title only)
  "tmdb_backdrop"      — scene still
  "tmdb_cast"          — cast headshot (names from KNOWN CAST only)
  "pexels_video"       — Pexels cinematic B-roll video (specific query)
  "pexels_image"       — Pexels still image (specific query)
  "text_card"          — styled text graphic
  "stat_card"          — big-number card
  "chapter_card"       — act transition (max 3 total)

Return ONLY a valid JSON array. Schema per item:
{{
  "segment_index": 0,
  "type": "user_upload_video",
  "upload_filename": "",
  "duration_seconds": 20,
  "trailer_timestamp": 0,
  "pexels_query": "",
  "cast_name": "",
  "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "",
  "wiki_title": "",
  "code_snippet": "", "code_language": "",
  "note": ""
}}

Rules:
- upload_filename: exact filename or "".
- pexels_query: 4-6 words SPECIFIC to this film's world (not genre labels).
- cast_name: exact name from KNOWN CAST only.
- Prioritise user_upload types first, then trailer_clip (30-50%), then TMDB/Pexels.
- Mix: at least 2 text/stat cards, 1 chapter_card.
- Sum ≈ word_count / 130 * 60 seconds.
"""


def _shot_prompt_tech(topic, matrix, script_outline, script_text, is_shorts, media_analysis):
    outline   = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    media_blk = _media_analysis_block(media_analysis)

    if is_shorts:
        return f"""
You are a video editor cutting a YouTube SHORT (9:16, under 60 seconds).

TOPIC: {topic}
SCRIPT: {script_text}

{media_blk}

Produce 3-6 segments, 5-12 seconds each.
Types: "user_upload_video", "user_upload_image", "pexels_video", "pexels_image",
       "text_card", "stat_card" (max 1 each of text/stat).

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_video", "upload_filename": "",
  "duration_seconds": 8, "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "",
  "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules: upload_filename = exact filename or "". pexels_query 4-6 specific words. Sum 30-58s.
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

{media_blk}

Produce 8-20 segments (15-35s each). Types:
  "user_upload_video"  — user clip (HIGHEST PRIORITY)
  "user_upload_image"  — user image (HIGHEST PRIORITY)
  "pexels_video"       — Pexels cinematic video (specific query)
  "pexels_image"       — Pexels still image (specific query)
  "wiki_image"         — Wikipedia image for company/product/person
  "text_card"          — headline, quote, timeline event
  "stat_card"          — big-number impact card
  "chapter_card"       — section transition (max 3 total)

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_video", "upload_filename": "",
  "duration_seconds": 20, "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "",
  "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules:
- upload_filename = exact filename or "".
- pexels_query: 4-6 specific words tied to this exact topic.
- wiki_title: exact Wikipedia article title.
- Mix: at least 40% pexels/upload, 2 stat_cards, 1 chapter_card.
- Sum ≈ word_count / 130 * 60 seconds.
"""


def _shot_prompt_edu(topic, matrix, script_outline, script_text, is_shorts, media_analysis):
    outline   = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    media_blk = _media_analysis_block(media_analysis)

    if is_shorts:
        return f"""
You are a video editor cutting a YouTube SHORT (9:16, under 60 seconds).

TOPIC: {topic}
SCRIPT: {script_text}

{media_blk}

Produce 3-6 segments, 5-12 seconds each.
Types: "user_upload_video", "user_upload_image", "pexels_video", "pexels_image",
       "text_card", "code_card" (max 1 code_card).

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_video", "upload_filename": "",
  "duration_seconds": 8, "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "",
  "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules: upload_filename = exact filename or "". pexels_query 4-6 specific words. Sum 30-58s.
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

{media_blk}

Produce 8-20 segments (15-35s each). Types:
  "user_upload_video"  — user clip (HIGHEST PRIORITY)
  "user_upload_image"  — user image (HIGHEST PRIORITY)
  "pexels_video"       — Pexels concept video (specific query)
  "pexels_image"       — Pexels still (specific query)
  "wiki_image"         — Wikipedia image for concept/person/tool
  "text_card"          — definition, key term, takeaway
  "code_card"          — code snippet (only when script references code)
  "stat_card"          — big-number fact
  "chapter_card"       — section transition (max 3 total)

Return ONLY a valid JSON array. Schema:
{{
  "segment_index": 0, "type": "pexels_video", "upload_filename": "",
  "duration_seconds": 20, "trailer_timestamp": 0,
  "pexels_query": "", "cast_name": "",
  "text_line1": "", "text_line2": "",
  "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "",
  "wiki_title": "", "code_snippet": "", "code_language": "", "note": ""
}}
Rules:
- upload_filename = exact filename or "".
- pexels_query: 4-6 specific words for this concept.
- Mix: at least 30% upload/pexels, 2 text/stat cards, 1 chapter_card.
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
    media_analysis = st.session_state.get("media_analysis", [])

    if mode == MODE_FILM:
        char_names = [c["name"] for c in package.get("character_matrix", [])]
        prompt = _shot_prompt_film(topic, source_type, matrix, char_names,
                                   script_outline, script_text,
                                   has_trailer, is_shorts, media_analysis)
    elif mode == MODE_TECH:
        prompt = _shot_prompt_tech(topic, matrix, script_outline,
                                   script_text, is_shorts, media_analysis)
    else:
        prompt = _shot_prompt_edu(topic, matrix, script_outline,
                                  script_text, is_shorts, media_analysis)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
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
# CLIP RESOLVER  — per-clip credit logic
# ─────────────────────────────────────────────────────────────────────────────
#
# Credit rules (per spec):
#   user_upload_*  → ""                        (creator's own content, no credit)
#   pexels_video   → "Video: Pexels.com"
#   pexels_image   → "Photo: Pexels.com"
#   trailer_clip   → "Courtesy: {topic} Official Trailer"
#   wiki/openverse → "Image: Wikipedia / Wikimedia Commons" or "Image: OpenVerse CC"
#   *_card         → ""                        (generated, no third-party source)

def _resolve_clip(shot, mode,
                  used_urls: set, topic,
                  trailer_path, trailer_segs,
                  trailer_idx,
                  cw: int, ch: int,
                  is_shorts: bool,
                  pexels_key: str,
                  media_analysis: list[dict]):
    """
    Resolve one shot dict → (visual, credit_str).
    visual      → np.ndarray (still) or MoviePy clip
    credit_str  → per-source attribution or ""
    """
    stype       = shot.get("type", "text_card")
    upload_fn   = shot.get("upload_filename", "")
    arr         = None
    clip_credit = ""

    # Build lookup: filename → path from media_analysis
    upload_map = {m["filename"]: m["path"] for m in media_analysis}

    def _fetch_unique(url):
        if not url or url in used_urls:
            return None
        a = _fetch_array(url, cw, ch)
        if a is not None:
            used_urls.add(url)
        return a

    # ── USER UPLOAD — VIDEO ─────────────────────────────────────────────────
    if stype == "user_upload_video":
        path = upload_map.get(upload_fn)
        if path and os.path.exists(path):
            try:
                duration = max(float(shot.get("duration_seconds", 12)), 5.0)
                clip = _loop_video_clip(path, duration, cw, ch)
                return clip, ""   # No credit for own content
            except Exception:
                pass  # fall through to fallback

    # ── USER UPLOAD — IMAGE ─────────────────────────────────────────────────
    elif stype == "user_upload_image":
        path = upload_map.get(upload_fn)
        if path and os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                arr = _fit_to_canvas(img, cw, ch)
                return arr, ""   # No credit for own content
            except Exception:
                pass

    # ── TRAILER CLIP ─────────────────────────────────────────────────────────
    elif stype == "trailer_clip":
        if trailer_path and trailer_segs:
            idx  = trailer_idx[0] % len(trailer_segs)
            s, e = trailer_segs[idx]
            trailer_idx[0] += 1
            try:
                vc = VideoFileClip(trailer_path).subclip(s, e)
                if vc.w != cw or vc.h != ch:
                    src_ratio = vc.w / vc.h
                    tgt_ratio = cw / ch
                    both_l = src_ratio >= 1.0 and tgt_ratio >= 1.0
                    both_p = src_ratio <  1.0 and tgt_ratio <  1.0
                    if both_l or both_p:
                        scale = max(cw / vc.w, ch / vc.h)
                        vc = vc.resize(scale).crop(
                            x_center=vc.w * scale / 2,
                            y_center=vc.h * scale / 2,
                            width=cw, height=ch)
                    else:
                        scale = min(cw / vc.w, ch / vc.h)
                        vc_r  = vc.resize(scale)
                        bg_arr = _fit_to_canvas(Image.fromarray(vc.get_frame(0)), cw, ch)
                        bg_clip = VideoClip(lambda t, b=bg_arr: b, duration=vc.duration)
                        from moviepy.editor import CompositeVideoClip
                        ox = (cw - vc_r.w) // 2
                        oy = (ch - vc_r.h) // 2
                        vc = CompositeVideoClip(
                            [bg_clip, vc_r.set_position((ox, oy))], size=(cw, ch))
                return vc, f"Courtesy: {topic} Official Trailer"
            except Exception:
                pass

    # ── PEXELS VIDEO ─────────────────────────────────────────────────────────
    elif stype == "pexels_video":
        query = shot.get("pexels_query") or topic
        orientation = "portrait" if is_shorts else "landscape"
        video_path = _pexels_video_download(
            query, pexels_key, PEXELS_CACHE_DIR, orientation)
        if video_path:
            try:
                duration = max(float(shot.get("duration_seconds", 12)), 5.0)
                clip = _loop_video_clip(video_path, duration, cw, ch)
                return clip, "Video: Pexels.com"
            except Exception:
                pass

    # ── PEXELS IMAGE ─────────────────────────────────────────────────────────
    elif stype == "pexels_image":
        query = shot.get("pexels_query") or topic
        orientation = "portrait" if is_shorts else "landscape"
        url = _pexels_image_url(query, pexels_key, orientation)
        arr = _fetch_unique(url)
        if arr is not None:
            clip_credit = "Photo: Pexels.com"

    # ── WIKI / TMDB (via Wikipedia) ──────────────────────────────────────────
    elif stype in ("wiki_image", "tmdb_poster", "tmdb_backdrop", "tmdb_cast"):
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
            clip_credit = "Image: Wikipedia / Wikimedia Commons"

    # ── OPENVERSE (generic B-roll fallback when no pexels key) ───────────────
    elif stype in ("pexels_broll", "openverse_image"):
        query = (shot.get("pexels_query") or shot.get("image_query") or topic)
        # Try Pexels image first if key available
        if pexels_key:
            url = _pexels_image_url(query, pexels_key)
            arr = _fetch_unique(url)
            if arr is not None:
                clip_credit = "Photo: Pexels.com"
        if arr is None:
            for page in range(1, 4):
                url = _openverse_image(query, page=page)
                arr = _fetch_unique(url)
                if arr is not None:
                    clip_credit = "Image: OpenVerse (CC licensed)"
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
    # Reached only if the primary resolver above didn't return early
    if arr is None and not hasattr(arr, "duration"):
        # 1. Pexels image on topic
        if pexels_key:
            url = _pexels_image_url(topic, pexels_key)
            arr = _fetch_unique(url)
            if arr is not None:
                clip_credit = "Photo: Pexels.com"
        # 2. OpenVerse
        if arr is None:
            url = _openverse_image(topic + " cinematic")
            arr = _fetch_unique(url)
            if arr is not None:
                clip_credit = "Image: OpenVerse (CC licensed)"
        # 3. Wikipedia
        if arr is None:
            url = _wiki_image(topic)
            arr = _fetch_unique(url)
            if arr is not None:
                clip_credit = "Image: Wikipedia / Wikimedia Commons"
        # 4. Generated text card — always succeeds
        if arr is None:
            arr = make_text_card(shot.get("note", topic), mode=mode, cw=cw, ch=ch)
            clip_credit = ""

    return arr, clip_credit


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ASSEMBLY WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _assembly_worker(audio_path, shot_list, mode, topic,
                     output_path, is_shorts, credit_override, pexels_key,
                     media_analysis):
    try:
        cw, ch = _canvas(is_shorts)

        # ── Trailer (film mode only) ──────────────────────────────────────────
        trailer_path = None
        trailer_segs = []
        if mode == MODE_FILM:
            _write_progress(2, "Searching for official trailer…")
            trailer_path = _download_trailer(topic)
            if trailer_path:
                n_segs = 4 if is_shorts else 8
                trailer_segs = _extract_trailer_segments(
                    trailer_path, num_segments=n_segs, seg_duration=8.0)
                _write_progress(6, f"Trailer ready — {len(trailer_segs)} segments")
            else:
                _write_progress(6, "Trailer not found — using stills/Pexels only")

        # ── Build clips ───────────────────────────────────────────────────────
        total       = len(shot_list)
        used_urls   = set()
        trailer_idx = [0]
        clips       = []
        clip_credits = []
        clip_starts  = []
        running_time = 0.0

        for i, shot in enumerate(shot_list):
            pct = 12 + int(68 * i / total)
            _write_progress(pct,
                f"Building clip {i+1}/{total} [{shot.get('type','?')}] — "
                f"{shot.get('note', shot.get('upload_filename', ''))}")

            duration = max(float(shot.get("duration_seconds", 12)), 5.0)
            zoom_in  = (i % 2 == 0)

            visual, auto_credit = _resolve_clip(
                shot, mode,
                used_urls, topic,
                trailer_path, trailer_segs,
                trailer_idx, cw, ch, is_shorts,
                pexels_key, media_analysis,
            )

            # visual is either an ndarray (still) or a MoviePy clip
            if hasattr(visual, "duration"):
                clip = visual.set_duration(min(visual.duration, duration))
            else:
                clip = _zoom_clip(visual, duration, zoom_in)

            clip_starts.append(running_time)
            clip_credits.append(auto_credit)
            running_time += duration
            clips.append(clip)

        # ── Stitch ────────────────────────────────────────────────────────────
        _write_progress(82, "Stitching clips…")
        video = concatenate_videoclips(clips, method="compose")

        # ── Audio ─────────────────────────────────────────────────────────────
        _write_progress(87, "Attaching voiceover…")
        audio = AudioFileClip(audio_path)

        if video.duration < audio.duration:
            extra      = audio.duration - video.duration + clips[-1].duration
            last_frame = clips[-1].get_frame(0)
            filler     = _zoom_clip(last_frame, extra, zoom_in=False)
            clips[-1]  = filler
            video      = concatenate_videoclips(clips, method="compose")

        _assembled = video.subclip(0, audio.duration).set_audio(audio)

        # ── Credit overlay ────────────────────────────────────────────────────
        _write_progress(91, "Adding per-clip credit overlay…")

        def _credit_at(t):
            # User override wins globally
            if credit_override:
                return credit_override
            # Otherwise use per-clip auto credit
            idx = bisect.bisect_right(clip_starts, t) - 1
            idx = max(0, min(idx, len(clip_credits) - 1))
            return clip_credits[idx]   # "" for user uploads and generated cards

        def _overlay_frame(t):
            f = _assembled.get_frame(t)
            return _add_credit_overlay(f, _credit_at(t), is_shorts, cw, ch)

        final = VideoClip(_overlay_frame, duration=_assembled.duration).set_audio(
            _assembled.audio)

        # ── Render ────────────────────────────────────────────────────────────
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
# CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_temp():
    """Remove keyframes and Pexels cache. Called on Reset All Steps."""
    import shutil
    for d in (KEYFRAME_DIR, PEXELS_CACHE_DIR):
        try:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎬 SudoVid")
st.caption("AI-Powered Script, Voice & Video Engine")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return ""

_gemini_from_secret = bool(_secret("GEMINI_API_KEY"))

with st.sidebar:
    st.header("🔑 API Keys")

    if _gemini_from_secret:
        api_key = _secret("GEMINI_API_KEY")
        st.success("✓ Gemini (from Secrets)")
    else:
        api_key = st.text_input(
            "Gemini API Key", type="password",
            help="Add GEMINI_API_KEY to Streamlit Secrets to hide this field")
        if api_key:
            st.success("✓ Gemini set")
        else:
            st.warning("⚠️ Gemini API Key required")

    st.markdown("---")
    pexels_key = st.text_input(
        "Pexels API Key",
        type="password",
        value=_secret("PEXELS_API_KEY"),
        help=(
            "Required for Pexels images AND Pexels video B-roll. "
            "Free at pexels.com/api — or add PEXELS_API_KEY to Streamlit Secrets."
        ),
    )
    if pexels_key:
        st.success("✓ Pexels set — images & videos enabled")
    else:
        st.info("ℹ️ Without Pexels key, B-roll falls back to OpenVerse CC images")

    st.divider()
    if st.button("Reset All Steps"):
        _cleanup_temp()
        st.session_state.clear()
        st.rerun()

    st.caption(f"Model: `{GEMINI_MODEL}`")

# ─────────────────────────────────────────────────────────────────────────────
# TABS  (6 tabs — Research & Script merged; YouTube Bundle moved last as optional)
# ─────────────────────────────────────────────────────────────────────────────
(tab1, tab2, tab3, tab4, tab5, tab6) = st.tabs([
    "1. Parameters",
    "2. Media Upload",
    "3. Research & Script",
    "4. Voiceover",
    "5. Video Assembly",
    "6. YouTube Bundle ✦ Optional",
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
            st.success("✅ Parameters saved! Click **'2. Media Upload'**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MEDIA UPLOAD  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Step 2: Upload Your Media")
    st.info(
        "Upload your own video clips and images. Gemini will analyse them and "
        "the shot list will prefer your media over any external source. "
        "You can skip this step and the app will source visuals from Pexels and OpenVerse."
    )

    st.markdown("#### 🎬 Video Clips")
    st.caption(
        f"Supported: {', '.join(f'.{e}' for e in ALLOWED_VIDEO_TYPES)} — "
        f"max **500 MB per file** — multiple files allowed"
    )
    uploaded_videos = st.file_uploader(
        "Upload video clips",
        type=ALLOWED_VIDEO_TYPES,
        accept_multiple_files=True,
        key="video_uploader",
        label_visibility="collapsed",
    )

    st.markdown("#### 🖼️ Images")
    st.caption(
        f"Supported: {', '.join(f'.{e}' for e in ALLOWED_IMAGE_TYPES)} — "
        f"max **50 MB per file** — multiple files allowed"
    )
    uploaded_images = st.file_uploader(
        "Upload images",
        type=ALLOWED_IMAGE_TYPES,
        accept_multiple_files=True,
        key="image_uploader",
        label_visibility="collapsed",
    )

    # Save & validate on change
    all_media_meta = st.session_state.get("uploaded_media", [])

    if st.button("💾 Save & Analyse Media"):
        if not api_key:
            st.warning("⚠️ Gemini API Key required for analysis.")
        else:
            saved = []
            for uf in (uploaded_videos or []):
                meta = _save_upload(uf, "video")
                if meta:
                    saved.append(meta)
            for uf in (uploaded_images or []):
                meta = _save_upload(uf, "image")
                if meta:
                    saved.append(meta)

            if saved:
                st.session_state["uploaded_media"] = saved
                with st.spinner(
                    f"🔍 Analysing {len(saved)} file(s) with Gemini Vision…"
                ):
                    analysis = analyse_uploaded_media(api_key, saved)
                    st.session_state["media_analysis"] = analysis
                st.success(f"✅ {len(analysis)} file(s) analysed and ready.")
            else:
                st.session_state["uploaded_media"]  = []
                st.session_state["media_analysis"]  = []
                st.info("No files saved. Proceeding without user media.")

    # Show analysis results
    analysis = st.session_state.get("media_analysis", [])
    if analysis:
        st.markdown("### 📋 Media Analysis Results")
        for m in analysis:
            icon = "🎬" if m.get("media_type") == "video" else "🖼️"
            tags = ", ".join(m.get("content_tags", []))
            st.markdown(
                f'<div class="media-card">'
                f'{icon} <b>{m["filename"]}</b> ({m["size_mb"]} MB)<br>'
                f'<b>Subjects:</b> {m.get("dominant_subjects", "—")}<br>'
                f'<b>Mood:</b> {m.get("mood", "—")} &nbsp;|&nbsp; '
                f'<b>Suggested use:</b> {m.get("suggested_use", "—")}<br>'
                f'<b>Tags:</b> {tags if tags else "—"}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.success("🎉 **Step 2 Complete!** Click **'3. Research & Script'** to continue.")
    elif st.session_state.get("uploaded_media") == []:
        st.success("Skipped — proceeding without user media. Click **'3. Research & Script'**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RESEARCH & SCRIPT  (merged)
# Two sequential buttons: Research first, then Generate Script.
# Research runs in the background; the briefing is shown collapsed so it
# doesn't block the eye path to the Generate Script button below it.
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Step 3: Research & Script")

    if "topic_param" not in st.session_state:
        st.info("Complete Step 1 first.")
    else:
        media_analysis_t3 = st.session_state.get("media_analysis", [])
        if media_analysis_t3:
            st.caption(
                f"💡 {len(media_analysis_t3)} uploaded media file(s) will inform "
                f"both research targeting and script structure."
            )

        # ── PHASE 1: RESEARCH ────────────────────────────────────────────────
        st.markdown("#### Phase 1 — Research")
        st.caption(
            "Gemini searches the web for facts, dates, quotes, and data that "
            "specifically support your angle. Takes 15–30 seconds."
        )

        if st.button("🔍 Research Topic", key="btn_research"):
            if not api_key:
                st.warning("Gemini API Key required in sidebar.")
            else:
                with st.spinner(
                    f"🌐 Researching **{st.session_state['topic_param']}**…"
                ):
                    st.session_state["research"] = perform_grounded_research(
                        topic       = st.session_state["topic_param"],
                        mode        = st.session_state["mode_param"],
                        source_type = st.session_state["source_param"],
                        angle       = st.session_state["angle_param"],
                        length      = st.session_state["length_param"],
                        api_key     = api_key,
                    )
                # Clear any stale script so Generate Script must be re-run
                st.session_state.pop("package", None)
                st.session_state.pop("final_script_text", None)

        if "research" in st.session_state:
            st.success("✅ Research complete")
            # Collapsed by default — inspectable but not blocking
            with st.expander("📄 View Factual Briefing", expanded=False):
                st.markdown(st.session_state["research"])

            st.markdown("---")

            # ── PHASE 2: GENERATE SCRIPT ─────────────────────────────────────
            st.markdown("#### Phase 2 — Generate Script")
            st.caption(
                "Gemini synthesises the research and your angle into a full "
                "conversational script. Edit it freely before moving to voiceover."
            )

            if st.button("📝 Generate Script", key="btn_generate_script"):
                with st.spinner(
                    f"✍️ Writing script for "
                    f"**{st.session_state.get('length_param', '')}**…"
                ):
                    st.session_state["package"] = generate_script_package(
                        mode           = st.session_state["mode_param"],
                        topic          = st.session_state["topic_param"],
                        research       = st.session_state["research"],
                        angle          = st.session_state["angle_param"],
                        matrix         = st.session_state["matrix_param"],
                        source_type    = st.session_state["source_param"],
                        length         = st.session_state["length_param"],
                        api_key        = api_key,
                        media_analysis = media_analysis_t3,
                    )

            if "package" in st.session_state:
                p = st.session_state["package"]
                if "error" in p:
                    st.error(p["error"])
                    with st.expander("Raw Output"):
                        st.text(p.get("raw"))
                else:
                    st.success(f"### {p.get('viral_title')}")

                    # Architecture details — collapsed, optional reading
                    with st.expander("📊 Script Architecture Details", expanded=False):
                        st.markdown("#### 🌍 Thematic Resonance")
                        st.warning(
                            f"**Analogous Event:** "
                            f"{p.get('thematic_resonance', {}).get('real_world_event')}"
                        )
                        st.write(p.get("thematic_resonance", {}).get("explanation"))
                        if st.session_state["mode_param"] == MODE_FILM:
                            for char in p.get("character_matrix", []):
                                st.markdown(
                                    f"**{char['name']}** "
                                    f"<span class='metric-badge'>"
                                    f"{char['arc_score']}/10</span>",
                                    unsafe_allow_html=True,
                                )

                    st.markdown("### 📝 Script Editor")
                    st.info(
                        "💡 Edit freely below — this is exactly what flows into "
                        "Voiceover. Use commas or `---` for natural pauses."
                    )
                    fs = p.get("full_script", {})
                    default_text = "\n\n".join(filter(None, [
                        p.get("hook_script", ""),
                        fs.get("intro", ""), fs.get("act1", ""),
                        fs.get("act2", ""), fs.get("act3", ""),
                        fs.get("outro", ""),
                    ]))
                    st.session_state["final_script_text"] = st.text_area(
                        "Final Polish:",
                        value=default_text.strip(),
                        height=400,
                    )
                    st.download_button(
                        "📥 Download Script (.txt)",
                        data=st.session_state["final_script_text"],
                        file_name=(
                            f"{p.get('viral_title','script').replace(' ','_').lower()}.txt"
                        ),
                        mime="text/plain",
                    )
                    st.success(
                        "🎉 **Step 3 Complete!** Click **'4. Voiceover'** to continue."
                    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — VOICEOVER
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Step 4: AI Voiceover Studio")
    st.info("Turn your finalised script into professional audio.")

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

    source_mode = st.radio("Choose Text Source for Voiceover:",
                            ["Use Generated Script (from Tab 3)",
                             "Upload Custom Text File (.txt)"])
    text_to_synthesize = ""
    if source_mode == "Use Generated Script (from Tab 3)":
        text_to_synthesize = st.session_state.get("final_script_text", "")
        if not text_to_synthesize:
            st.warning("⚠️ No generated script found. Complete Step 3 first.")
    else:
        uploaded_file = st.file_uploader("Upload .txt for Voiceover",
                                          type=["txt"], key="voice_upload")
        if uploaded_file:
            text_to_synthesize = uploaded_file.getvalue().decode("utf-8")
            st.success("File uploaded!")

    st.markdown("### Preview Text for Audio Generation")
    st.session_state["tab4_audio_text"] = st.text_area(
        "This exact text will be sent to the AI Voice:",
        value=text_to_synthesize, height=250)

    if st.button("🔊 Generate Voiceover"):
        if not st.session_state.get("tab4_audio_text", "").strip():
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
                            file_name=(
                                f"{st.session_state.get('topic_param','voiceover').replace(' ','_').lower()}"
                                f"_voiceover.mp3"
                            ),
                            mime="audio/mp3",
                        )
                else:
                    st.error("Audio generation failed. Check internet connection.")
            st.success("🎉 **Step 4 Complete!** Click **'5. Video Assembly'** to continue.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — VIDEO ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Step 5: Video Assembly")

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

    # Media summary
    media_analysis = st.session_state.get("media_analysis", [])
    if media_analysis:
        n_vid = sum(1 for m in media_analysis if m.get("media_type") == "video")
        n_img = sum(1 for m in media_analysis if m.get("media_type") == "image")
        st.success(
            f"📁 **{len(media_analysis)} uploaded files** ready "
            f"({n_vid} videos, {n_img} images) — will be preferred in shot list."
        )
    else:
        st.caption("No user media uploaded — visuals sourced from Pexels and OpenVerse.")

    # Pexels status in context of assembly
    if pexels_key:
        st.caption("✅ Pexels enabled — images AND video B-roll available (highest quality)")
    else:
        st.caption("⚠️ No Pexels key — B-roll will use OpenVerse CC images only")

    # Credit override
    override_on = st.checkbox(
        "✏️ Override all automatic credits with a custom global credit",
        value=False,
        help=(
            "By default, credits are set per clip:\n"
            "• User uploads → no credit\n"
            "• Pexels video → 'Video: Pexels.com'\n"
            "• Pexels image → 'Photo: Pexels.com'\n"
            "• Trailer → 'Courtesy: [Topic] Official Trailer'\n"
            "• Wikipedia → 'Image: Wikipedia / Wikimedia Commons'\n"
            "• Generated cards → no credit\n\n"
            "Enable this only if you want a single credit for the whole video."
        ),
    )
    credit_override = ""
    if override_on:
        credit_override = st.text_input(
            "Custom global credit text",
            value=f"Courtesy: {topic_val}" if topic_val else "Courtesy: Studio Name",
        )
    else:
        st.caption(
            "🎯 Per-clip credits: your uploads show no credit; "
            "Pexels/Wikipedia/trailer show their own source."
        )

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
        # Film-mode trailer pre-fetch
        if mode == MODE_FILM:
            st.caption(
                "ℹ️ Trailers sourced from hd-trailers.net → Apple CDN, "
                "Apple XML feed, IMDb, and Internet Archive — no extra keys needed."
            )
            if st.button("🎞️ Pre-fetch Trailer (optional but recommended)"):
                with st.spinner(f"Fetching {topic_val} trailer…"):
                    tp = _download_trailer(topic_val)
                    if tp:
                        segs = _extract_trailer_segments(tp)
                        st.session_state["trailer_path"] = tp
                        st.session_state["trailer_segs"] = segs
                        st.success(f"✅ Trailer ready — {len(segs)} segments")
                    else:
                        st.session_state["trailer_path"] = None
                        st.session_state["trailer_segs"] = []
                        st.warning(
                            "Trailer not found. Assembly will use user uploads + Pexels/OpenVerse."
                        )

        # Shot list generation
        if st.button("🎬 Generate Shot List"):
            trailer_available = bool(st.session_state.get("trailer_path"))
            with st.spinner("Gemini is planning your shot list…"):
                st.session_state["shot_list"] = generate_shot_list(
                    api_key, has_trailer=trailer_available, is_shorts=is_shorts)

        if "shot_list" in st.session_state:
            sl        = st.session_state["shot_list"]
            total_dur = sum(s.get("duration_seconds", 12) for s in sl)

            # Count how many shots use user uploads
            upload_shots = sum(
                1 for s in sl
                if s.get("type") in ("user_upload_video", "user_upload_image")
                   and s.get("upload_filename")
            )
            st.success(
                f"Shot list ready — **{len(sl)} clips**, ~**{int(total_dur)}s** total"
                + (f" | **{upload_shots} shots** use your uploaded media" if upload_shots else "")
            )

            type_icons = {
                "user_upload_video": "🎥",
                "user_upload_image": "🖼️",
                "trailer_clip":      "🎬",
                "wiki_image":        "📖",
                "tmdb_poster":       "🖼️",
                "tmdb_backdrop":     "🎞️",
                "tmdb_cast":         "👤",
                "pexels_video":      "📹",
                "pexels_image":      "🌆",
                "pexels_broll":      "🌆",
                "openverse_image":   "🌆",
                "text_card":         "📝",
                "stat_card":         "📊",
                "code_card":         "💻",
                "chapter_card":      "📌",
            }

            with st.expander("📋 View Shot List", expanded=False):
                for s in sl:
                    icon   = type_icons.get(s.get("type", ""), "▪️")
                    upload = f" [{s.get('upload_filename')}]" if s.get("upload_filename") else ""
                    detail = (s.get("pexels_query") or s.get("cast_name") or
                              s.get("wiki_title")  or s.get("text_line1") or
                              s.get("stat_value")  or s.get("chapter_title") or
                              s.get("note", ""))
                    st.markdown(
                        f"`{s.get('segment_index', 0):02d}` {icon} "
                        f"**{s.get('type','')}**{upload} — "
                        f"{s.get('duration_seconds','?')}s — _{detail}_"
                    )

            st.markdown("---")
            output_path = os.path.join(TMP_ROOT, "sa_output_video.mp4")

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
                            pexels_key      = pexels_key,
                            media_analysis  = media_analysis,
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
                                file_name=(
                                    f"{topic_val.replace(' ','_').lower()}"
                                    f"{'_shorts' if is_shorts else ''}_video.mp4"
                                ),
                                mime="video/mp4",
                            )
                    st.info(
                        "💡 Head to **Tab 6 — YouTube Bundle** whenever you're "
                        "ready to generate your title, description, tags, and "
                        "thumbnail prompt. It's optional and can be done any time."
                    )
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
# TAB 6 — YOUTUBE BUNDLE  (optional post-production step)
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Step 6: YouTube Bundle")
    st.info(
        "**This step is optional and can be completed at any time** — before or "
        "after video assembly. Generate your YouTube title, description, tags, "
        "hashtags, and an AI thumbnail prompt ready to paste into YouTube Studio."
    )

    # Source selector — references updated tab numbers
    bundle_source = st.radio(
        "Base the bundle on:",
        [
            "Generated script (Tab 3)",
            "Final voiceover text (Tab 4)",
        ],
    )
    if st.button("📦 Generate YouTube Bundle"):
        target_text = (
            st.session_state.get("final_script_text", "")
            if bundle_source == "Generated script (Tab 3)"
            else st.session_state.get("tab4_audio_text", "")
        )
        if not api_key:
            st.error("⚠️ Gemini API Key required.")
        elif not target_text.strip():
            st.error(
                "⚠️ No script text found. "
                "Complete Step 3 (Research & Script) first."
            )
        else:
            with st.spinner("Generating YouTube metadata…"):
                st.session_state["yt_bundle"] = generate_youtube_bundle(
                    api_key, target_text)

    if "yt_bundle" in st.session_state:
        bundle = st.session_state["yt_bundle"]
        if "error" in bundle:
            st.error(bundle["error"])
        else:
            st.success("✅ YouTube Bundle Generated!")
            st.markdown("### 📝 Metadata")
            st.text_input("Viral Title",  value=bundle.get("viral_title",  ""))
            st.text_area( "Description",  value=bundle.get("description",  ""), height=200)
            c1, c2 = st.columns(2)
            with c1:
                st.text_area("Tags",     value=", ".join(bundle.get("tags",     [])), height=100)
            with c2:
                st.text_area("Hashtags", value=" ".join(bundle.get("hashtags", [])), height=100)
            st.markdown("---")
            st.markdown("### 🎨 AI Thumbnail Prompt")
            st.caption("Paste into Midjourney, DALL-E, or Canva.")
            st.text_area(
                "Image Prompt:",
                value=bundle.get("thumbnail_prompt", ""),
                height=120,
            )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"SudoVid v3.1 | AI-Powered Script, Voice & Video Engine | "
    f"Model: {GEMINI_MODEL} | 6-tab flow — YouTube Bundle optional"
)