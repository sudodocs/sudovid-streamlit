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
# CSS  — Phase 1: Foundation overhaul
# Replace the existing st.markdown("""<style>…</style>""") block with this.
# Zero logic changes — purely cosmetic layer.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── DESIGN TOKENS ──────────────────────────────────────────────────────── */
:root {
    /* Brand palette */
    --sv-blue:        #2563eb;
    --sv-blue-dark:   #1d4ed8;
    --sv-blue-dim:    #eff6ff;
    --sv-blue-border: #bfdbfe;

    /* Neutrals */
    --sv-bg:          #f8fafc;
    --sv-surface:     #ffffff;
    --sv-surface-2:   #f1f5f9;
    --sv-border:      #e2e8f0;
    --sv-border-med:  #cbd5e1;

    /* Text */
    --sv-text:        #0f172a;
    --sv-text-2:      #475569;
    --sv-text-3:      #94a3b8;

    /* Status */
    --sv-green:       #16a34a;
    --sv-green-dim:   #f0fdf4;
    --sv-green-bdr:   #bbf7d0;
    --sv-amber:       #d97706;
    --sv-amber-dim:   #fffbeb;
    --sv-amber-bdr:   #fde68a;
    --sv-red:         #dc2626;
    --sv-red-dim:     #fef2f2;
    --sv-red-bdr:     #fecaca;

    /* Spacing */
    --sv-radius-sm:   6px;
    --sv-radius:      10px;
    --sv-radius-lg:   14px;
}

/* ── APP SHELL ──────────────────────────────────────────────────────────── */
.stApp {
    background-color: var(--sv-bg) !important;
    color: var(--sv-text) !important;
}

/* ── TYPOGRAPHY ─────────────────────────────────────────────────────────── */
p, span,
.stMarkdown, .stMarkdown p,
h1, h2, h3, h4,
.stMetric label {
    color: var(--sv-text) !important;
}
.stCaption, .stCaption p {
    color: var(--sv-text-2) !important;
    font-size: 0.82rem !important;
}

/* ── FORM INPUTS ─────────────────────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea {
    background-color: var(--sv-surface) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius-sm) !important;
    color: var(--sv-text) !important;
    font-size: 0.92rem !important;
    transition: border-color 0.15s ease;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--sv-blue) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.08) !important;
    outline: none !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: var(--sv-text-3) !important;
}

/* Select / Multiselect */
[data-baseweb="select"],
.stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--sv-surface) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius-sm) !important;
    color: var(--sv-text) !important;
}
[data-baseweb="select"]:focus-within {
    border-color: var(--sv-blue) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.08) !important;
}

/* Select dropdown menu */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background-color: var(--sv-surface) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius) !important;
    box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.12) !important;
}
[data-baseweb="menu"] li:hover {
    background-color: var(--sv-blue-dim) !important;
    color: var(--sv-blue) !important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: var(--sv-blue) !important;
    border-color: var(--sv-blue) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"],
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background-color: var(--sv-border) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
    background-color: var(--sv-blue) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--sv-border-med) !important;
    border-radius: var(--sv-radius) !important;
    background-color: var(--sv-surface) !important;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--sv-blue) !important;
    background-color: var(--sv-blue-dim) !important;
}

/* Radio */
[data-testid="stRadio"] label {
    cursor: pointer !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
}

/* Checkbox */
[data-baseweb="checkbox"] [data-testid="stCheckbox"] span {
    border-radius: 4px !important;
}

/* ── BUTTONS ─────────────────────────────────────────────────────────────── */

/* Primary (default) */
.stButton > button {
    background-color: var(--sv-blue) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--sv-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    height: 2.75rem !important;
    padding: 0 1.25rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: background-color 0.15s ease, transform 0.1s ease,
                box-shadow 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06),
                0 1px 0 rgba(255,255,255,0.08) inset !important;
}
.stButton > button:hover {
    background-color: var(--sv-blue-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: var(--sv-green) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--sv-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    height: 2.75rem !important;
    padding: 0 1.25rem !important;
    width: 100% !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06) !important;
}
.stDownloadButton > button:hover {
    background-color: #15803d !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.25) !important;
}

/* ── TABS ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid var(--sv-border) !important;
    background-color: transparent !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    height: 44px !important;
    background-color: transparent !important;
    color: var(--sv-text-2) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0 16px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
    white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--sv-blue) !important;
    background-color: var(--sv-blue-dim) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--sv-blue) !important;
    border-bottom-color: var(--sv-blue) !important;
    font-weight: 600 !important;
}
/* Hide the default tab focus ring — we have our own hover */
.stTabs [data-baseweb="tab"]:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* ── METRICS ─────────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    color: var(--sv-blue) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--sv-text-2) !important;
    font-size: 0.8rem !important;
}
[data-testid="stMetric"] {
    background-color: var(--sv-surface) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius) !important;
    padding: 16px 20px !important;
}

/* ── ALERTS / STATUS BANNERS ────────────────────────────────────────────── */
/* Success */
[data-testid="stAlert"][kind="success"],
.stAlert.success,
div[data-baseweb="notification"][kind="positive"] {
    background-color: var(--sv-green-dim) !important;
    border: 1px solid var(--sv-green-bdr) !important;
    border-radius: var(--sv-radius) !important;
    color: #14532d !important;
}
/* Warning */
[data-testid="stAlert"][kind="warning"],
.stAlert.warning,
div[data-baseweb="notification"][kind="warning"] {
    background-color: var(--sv-amber-dim) !important;
    border: 1px solid var(--sv-amber-bdr) !important;
    border-radius: var(--sv-radius) !important;
    color: #78350f !important;
}
/* Error */
[data-testid="stAlert"][kind="error"],
.stAlert.error,
div[data-baseweb="notification"][kind="negative"] {
    background-color: var(--sv-red-dim) !important;
    border: 1px solid var(--sv-red-bdr) !important;
    border-radius: var(--sv-radius) !important;
    color: #7f1d1d !important;
}
/* Info */
[data-testid="stAlert"][kind="info"],
.stAlert.info,
div[data-baseweb="notification"][kind="info"] {
    background-color: var(--sv-blue-dim) !important;
    border: 1px solid var(--sv-blue-border) !important;
    border-radius: var(--sv-radius) !important;
    color: #1e3a8a !important;
}

/* ── EXPANDERS ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius) !important;
    background-color: var(--sv-surface) !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--sv-text) !important;
    padding: 12px 16px !important;
    border-radius: var(--sv-radius) !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: var(--sv-surface-2) !important;
}
[data-testid="stExpander"][open] summary {
    border-bottom: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius) var(--sv-radius) 0 0 !important;
}

/* ── SPINNER / PROGRESS ──────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background-color: var(--sv-blue) !important;
    border-radius: 99px !important;
    transition: width 0.4s ease !important;
}
[data-testid="stProgress"] > div {
    background-color: var(--sv-border) !important;
    border-radius: 99px !important;
    height: 6px !important;
}

/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--sv-surface) !important;
    border-right: 1px solid var(--sv-border) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    color: var(--sv-red) !important;
    border: 1px solid var(--sv-red-bdr) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: var(--sv-red-dim) !important;
    transform: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--sv-border) !important;
    margin: 12px 0 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--sv-text-3) !important;
}

/* ── DIVIDER ─────────────────────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] hr,
.stDivider {
    border-color: var(--sv-border) !important;
    margin: 20px 0 !important;
}

/* ── CODE BLOCKS ─────────────────────────────────────────────────────────── */
.stCode, code, pre {
    background-color: var(--sv-surface-2) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius-sm) !important;
    font-size: 0.82rem !important;
}

/* ── AUDIO PLAYER ────────────────────────────────────────────────────────── */
audio {
    width: 100% !important;
    border-radius: var(--sv-radius-sm) !important;
    accent-color: var(--sv-blue) !important;
}

/* ── CUSTOM HTML COMPONENT CLASSES ──────────────────────────────────────── */

/* General-purpose card */
.sv-card {
    background-color: var(--sv-surface);
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius-lg);
    padding: 20px 22px;
    margin-bottom: 16px;
}

/* Card with blue left-accent — for active/info sections */
.sv-card-accent {
    background-color: var(--sv-surface);
    border: 1px solid var(--sv-border);
    border-left: 3px solid var(--sv-blue);
    border-radius: 0 var(--sv-radius-lg) var(--sv-radius-lg) 0;
    padding: 16px 20px;
    margin-bottom: 14px;
}

/* Success card */
.sv-card-success {
    background-color: var(--sv-green-dim);
    border: 1px solid var(--sv-green-bdr);
    border-radius: var(--sv-radius);
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #14532d;
}

/* Warning card */
.sv-card-warn {
    background-color: var(--sv-amber-dim);
    border: 1px solid var(--sv-amber-bdr);
    border-radius: var(--sv-radius);
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #78350f;
}

/* Tuning matrix container (wraps sliders) */
.sv-matrix {
    background-color: var(--sv-surface);
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius-lg);
    padding: 20px 22px;
    margin-bottom: 20px;
}

/* Media item card (used in Tab 2 analysis results) */
.sv-media-card {
    background-color: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: var(--sv-radius);
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--sv-text);
}

/* Pill / badge */
.sv-badge {
    display: inline-block;
    background-color: var(--sv-blue-dim);
    color: var(--sv-blue);
    border: 1px solid var(--sv-blue-border);
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.sv-badge-green {
    background-color: var(--sv-green-dim);
    color: var(--sv-green);
    border-color: var(--sv-green-bdr);
}
.sv-badge-amber {
    background-color: var(--sv-amber-dim);
    color: var(--sv-amber);
    border-color: var(--sv-amber-bdr);
}
.sv-badge-gray {
    background-color: var(--sv-surface-2);
    color: var(--sv-text-2);
    border-color: var(--sv-border);
}

/* Section header row (icon + title + optional badge inline) */
.sv-section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.sv-section-head h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--sv-text) !important;
}

/* Stat / metric row */
.sv-stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}
.sv-stat {
    flex: 1;
    background-color: var(--sv-surface-2);
    border-radius: var(--sv-radius);
    padding: 12px 14px;
    text-align: center;
}
.sv-stat-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--sv-blue);
    display: block;
}
.sv-stat-lbl {
    font-size: 0.75rem;
    color: var(--sv-text-2);
    display: block;
    margin-top: 2px;
}

/* Footer */
.sv-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 0 6px;
    border-top: 1px solid var(--sv-border);
    margin-top: 10px;
    font-size: 0.78rem;
    color: var(--sv-text-3);
    flex-wrap: wrap;
    gap: 8px;
}

/* ── LEGACY ALIASES (keep existing class names working) ─────────────────── */
/* Any existing .report-card and .media-card usages keep working */
.report-card {
    background-color: var(--sv-surface);
    padding: 20px 22px;
    border-radius: var(--sv-radius-lg);
    border: 1px solid var(--sv-border);
    margin-bottom: 20px;
}
.media-card {
    background-color: #f0f9ff;
    padding: 12px 14px;
    border-radius: var(--sv-radius);
    border: 1px solid #bae6fd;
    margin-bottom: 8px;
    font-size: 0.85em;
}
.metric-badge {
    background-color: var(--sv-blue-dim);
    color: #1e40af;
    border: 1px solid var(--sv-blue-border);
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85em;
}
/* ── HERO HEADER ─────────────────────────────────────────────────────────── */
.sv-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    background-color: var(--sv-surface);
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius-lg);
    padding: 16px 22px;
    margin-bottom: 20px;
}
.sv-hero-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.sv-hero-icon {
    font-size: 28px;
    line-height: 1;
}
.sv-hero-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.sv-hero-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--sv-text) !important;
    letter-spacing: -0.01em;
}
.sv-hero-sub {
    font-size: 0.78rem;
    color: var(--sv-text-2) !important;
}
 
/* ── STEP PROGRESS TRACKER ───────────────────────────────────────────────── */
.sv-step-track {
    display: flex;
    align-items: flex-start;
    gap: 0;
}
.sv-step-dot {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.sv-step-circle {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    cursor: default;
    transition: transform 0.15s ease;
}
.sv-step-circle:hover { transform: scale(1.12); }
 
.sv-step-done {
    background-color: var(--sv-green-dim);
    color: var(--sv-green) !important;
    border: 1.5px solid var(--sv-green-bdr);
}
.sv-step-active {
    background-color: var(--sv-blue-dim);
    color: var(--sv-blue) !important;
    border: 1.5px solid var(--sv-blue);
}
.sv-step-locked {
    background-color: var(--sv-surface-2);
    color: var(--sv-text-3) !important;
    border: 1px solid var(--sv-border);
}
.sv-step-label {
    font-size: 9px;
    color: var(--sv-text-3) !important;
    text-align: center;
    max-width: 40px;
    line-height: 1.2;
    white-space: nowrap;
}
.sv-step-connector {
    width: 18px;
    height: 1px;
    background-color: var(--sv-border);
    margin-top: 13px;   /* vertically center with circle mid-point */
    flex-shrink: 0;
}
 
/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
.sv-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0 12px;
    border-bottom: 1px solid var(--sv-border);
    margin-bottom: 14px;
}
.sv-sidebar-brand-icon { font-size: 22px; line-height: 1; }
.sv-sidebar-brand-name {
    font-size: 1rem;
    font-weight: 700;
    color: var(--sv-text) !important;
    letter-spacing: -0.01em;
}
.sv-sidebar-section-label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--sv-text-3) !important;
    margin: 0 0 8px !important;
}
 
/* API key status rows */
.sv-key-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: var(--sv-radius-sm);
    margin-bottom: 4px;
    font-size: 0.82rem;
}
.sv-key-ok   { background-color: var(--sv-green-dim); border: 1px solid var(--sv-green-bdr); }
.sv-key-missing { background-color: var(--sv-red-dim); border: 1px solid var(--sv-red-bdr); }
.sv-key-warn { background-color: var(--sv-amber-dim); border: 1px solid var(--sv-amber-bdr); }
 
.sv-key-icon { font-size: 0.9rem; width: 16px; text-align: center; flex-shrink: 0; }
.sv-key-name { font-weight: 600; color: var(--sv-text) !important; flex: 1; }
.sv-key-source { font-size: 0.74rem; color: var(--sv-text-2) !important; }
 
.sv-key-ok   .sv-key-icon { color: var(--sv-green) !important; }
.sv-key-missing .sv-key-icon { color: var(--sv-red) !important; }
.sv-key-warn .sv-key-icon { color: var(--sv-amber) !important; }
 
/* Sidebar workflow progress rows */
.sv-prog-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    border-radius: var(--sv-radius-sm);
    margin-bottom: 2px;
    font-size: 0.82rem;
}
.sv-prog-done   { background-color: var(--sv-green-dim); }
.sv-prog-active { background-color: var(--sv-blue-dim);  border-left: 2px solid var(--sv-blue); }
.sv-prog-locked { opacity: 0.5; }
 
.sv-prog-icon  { width: 14px; text-align: center; font-weight: 700; flex-shrink: 0; }
.sv-prog-label { color: var(--sv-text) !important; }
.sv-prog-done  .sv-prog-icon { color: var(--sv-green) !important; }
.sv-prog-active .sv-prog-icon { color: var(--sv-blue) !important; }
 
/* Model badge */
.sv-model-badge {
    font-size: 0.72rem;
    color: var(--sv-text-3) !important;
    text-align: center;
    padding: 6px 0 0;
    font-family: monospace;
}
 
/* ── FOOTER ──────────────────────────────────────────────────────────────── */
.sv-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    padding: 14px 0 6px;
    border-top: 1px solid var(--sv-border);
    margin-top: 20px;
    font-size: 0.76rem;
    color: var(--sv-text-3) !important;
}
.sv-footer code {
    font-size: 0.72rem;
    background-color: var(--sv-surface-2);
    border: 1px solid var(--sv-border);
    padding: 1px 5px;
    border-radius: 4px;
}
/* ── TAB 1: MODE SELECTOR CARDS ─────────────────────────────────────────── */
.sv-mode-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}
.sv-mode-card {
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius-lg);
    padding: 14px 12px;
    cursor: pointer;
    background: var(--sv-surface);
    text-align: center;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.sv-mode-card:hover {
    border-color: var(--sv-blue);
    background-color: var(--sv-blue-dim);
}
.sv-mode-card.sv-selected {
    border: 2px solid var(--sv-blue);
    background-color: var(--sv-blue-dim);
}
.sv-mode-icon { font-size: 22px; margin-bottom: 6px; }
.sv-mode-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--sv-text) !important;
    margin: 0 0 3px;
}
.sv-mode-hint {
    font-size: 0.72rem;
    color: var(--sv-text-2) !important;
    line-height: 1.4;
}
.sv-mode-card.sv-selected .sv-mode-name { color: var(--sv-blue) !important; }
.sv-mode-card.sv-selected .sv-mode-hint { color: var(--sv-blue) !important; opacity: 0.8; }
 
/* ── TAB 1: LENGTH CARDS ─────────────────────────────────────────────────── */
.sv-len-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}
.sv-len-card {
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius);
    padding: 10px 12px;
    background: var(--sv-surface);
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.sv-len-card.sv-selected {
    border: 2px solid var(--sv-blue);
    background-color: var(--sv-blue-dim);
}
.sv-len-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--sv-text) !important;
    margin: 0 0 4px;
    display: block;
}
.sv-len-badge {
    display: inline-block;
    font-size: 0.7rem;
    padding: 1px 7px;
    border-radius: 99px;
    background: var(--sv-surface-2);
    color: var(--sv-text-2) !important;
    border: 1px solid var(--sv-border);
}
.sv-len-card.sv-selected .sv-len-name { color: var(--sv-blue) !important; }
.sv-len-card.sv-selected .sv-len-badge {
    background: var(--sv-surface);
    color: var(--sv-blue) !important;
    border-color: var(--sv-blue-border);
}
 
/* ── TAB 1: ANGLE CHAR COUNTER ───────────────────────────────────────────── */
.sv-angle-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
    font-size: 0.75rem;
    color: var(--sv-text-3) !important;
}
.sv-char-count-warn { color: var(--sv-amber) !important; font-weight: 600; }
 
/* ── TAB 3: SCRIPT STATS BAR ─────────────────────────────────────────────── */
.sv-script-stats {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.sv-script-stat {
    background: var(--sv-surface-2);
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius-sm);
    padding: 5px 12px;
    font-size: 0.78rem;
    color: var(--sv-text-2) !important;
    display: flex;
    align-items: center;
    gap: 5px;
}
.sv-script-stat strong { color: var(--sv-blue) !important; font-weight: 700; }
 
/* ── TAB 4: VOICE CARDS ──────────────────────────────────────────────────── */
.sv-voice-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-bottom: 16px;
}
.sv-voice-card {
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius);
    padding: 10px 12px;
    cursor: pointer;
    background: var(--sv-surface);
    display: flex;
    align-items: center;
    gap: 10px;
    transition: border-color 0.15s ease;
}
.sv-voice-card:hover { border-color: var(--sv-blue); }
.sv-voice-card.sv-selected {
    border: 2px solid var(--sv-blue);
    background: var(--sv-blue-dim);
}
.sv-voice-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    background: var(--sv-surface-2);
}
.sv-voice-card.sv-selected .sv-voice-avatar { background: var(--sv-blue-border); }
.sv-voice-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--sv-text) !important;
    margin: 0 0 2px;
}
.sv-voice-card.sv-selected .sv-voice-name { color: var(--sv-blue) !important; }
.sv-voice-style {
    font-size: 0.7rem;
    color: var(--sv-text-2) !important;
    margin: 0;
}
.sv-voice-gender {
    margin-left: auto;
    font-size: 0.68rem;
    padding: 1px 7px;
    border-radius: 99px;
    border: 1px solid var(--sv-border);
    color: var(--sv-text-3) !important;
    background: var(--sv-surface-2);
    flex-shrink: 0;
}
 
/* ── TAB 5: PHASE PROGRESS ───────────────────────────────────────────────── */
.sv-phase-bar {
    display: flex;
    gap: 4px;
    margin-top: 8px;
    margin-bottom: 14px;
}
.sv-phase-pill {
    flex: 1;
    text-align: center;
    font-size: 0.68rem;
    padding: 4px 2px;
    border-radius: var(--sv-radius-sm);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sv-phase-done    { background: var(--sv-green-dim); color: var(--sv-green) !important; }
.sv-phase-active  { background: var(--sv-blue-dim);  color: var(--sv-blue)  !important; font-weight: 600; }
.sv-phase-pending { background: var(--sv-surface-2); color: var(--sv-text-3) !important; }
 
/* ── TAB 6: BUNDLE OUTPUT CARDS ──────────────────────────────────────────── */
.sv-bundle-card {
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius-lg);
    padding: 14px 16px;
    background: var(--sv-surface);
    margin-bottom: 12px;
}
.sv-bundle-card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sv-text-3) !important;
    margin: 0 0 8px;
}
.sv-bundle-title-text {
    font-size: 1rem;
    font-weight: 600;
    color: var(--sv-text) !important;
    margin: 0;
    line-height: 1.4;
}
.sv-thumbnail-card {
    border: 1px solid var(--sv-blue-border);
    border-radius: var(--sv-radius-lg);
    background: var(--sv-blue-dim);
    padding: 14px 16px;
    margin-bottom: 12px;
}
.sv-thumbnail-card .sv-bundle-card-label { color: var(--sv-blue) !important; }
 
/* ── STEP COMPLETE BANNER ────────────────────────────────────────────────── */
.sv-complete-banner {
    background: var(--sv-green-dim);
    border: 1px solid var(--sv-green-bdr);
    border-radius: var(--sv-radius);
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 14px;
    flex-wrap: wrap;
    gap: 8px;
}
.sv-complete-text {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--sv-green) !important;
}
.sv-complete-cta {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--sv-blue) !important;
    background: var(--sv-surface);
    border: 1px solid var(--sv-blue-border);
    border-radius: var(--sv-radius-sm);
    padding: 5px 12px;
    cursor: pointer;
    white-space: nowrap;
}
/* ── LOCKED TAB GATE ─────────────────────────────────────────────────────── */
.sv-locked-gate {
    border: 1px solid var(--sv-border);
    border-left: 3px solid var(--sv-border-med);
    border-radius: 0 var(--sv-radius-lg) var(--sv-radius-lg) 0;
    background: var(--sv-surface);
    padding: 20px 22px;
    margin-top: 8px;
}
.sv-locked-icon {
    font-size: 28px;
    margin-bottom: 10px;
    display: block;
}
.sv-locked-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--sv-text) !important;
    margin: 0 0 6px;
}
.sv-locked-body {
    font-size: 0.82rem;
    color: var(--sv-text-2) !important;
    margin: 0 0 14px;
    line-height: 1.6;
}
.sv-locked-steps {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.sv-locked-step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
}
.sv-locked-step-icon {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 700;
    flex-shrink: 0;
}
.sv-locked-step-done {
    background: var(--sv-green-dim);
    color: var(--sv-green) !important;
    border: 1px solid var(--sv-green-bdr);
}
.sv-locked-step-todo {
    background: var(--sv-surface-2);
    color: var(--sv-text-3) !important;
    border: 1px solid var(--sv-border);
}
.sv-locked-step-label-done {
    color: var(--sv-text-2) !important;
    text-decoration: line-through;
}
.sv-locked-step-label-todo {
    color: var(--sv-text) !important;
    font-weight: 500;
}
 
/* ── TAB 2: MEDIA CARD (upgraded from legacy .media-card) ───────────────── */
.sv-media-item {
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius);
    background: var(--sv-surface);
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.sv-media-item-icon {
    font-size: 20px;
    flex-shrink: 0;
    margin-top: 2px;
}
.sv-media-item-body { flex: 1; min-width: 0; }
.sv-media-item-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--sv-text) !important;
    margin: 0 0 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sv-media-item-row {
    font-size: 0.76rem;
    color: var(--sv-text-2) !important;
    margin: 0 0 2px;
    line-height: 1.5;
}
.sv-media-item-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
}
.sv-media-tag {
    font-size: 0.68rem;
    padding: 1px 7px;
    border-radius: 99px;
    background: var(--sv-surface-2);
    border: 1px solid var(--sv-border);
    color: var(--sv-text-2) !important;
}
 
/* ── TAB 5: PRE-FLIGHT CHECKLIST ────────────────────────────────────────── */
.sv-preflight {
    border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius-lg);
    background: var(--sv-surface);
    padding: 16px 18px;
    margin-bottom: 16px;
}
.sv-preflight-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sv-text-3) !important;
    margin: 0 0 10px;
}
.sv-preflight-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid var(--sv-border);
    font-size: 0.82rem;
}
.sv-preflight-row:last-child { border-bottom: none; }
.sv-preflight-ok   { color: var(--sv-green) !important; font-weight: 700; }
.sv-preflight-fail { color: var(--sv-red)   !important; font-weight: 700; }
.sv-preflight-label { color: var(--sv-text) !important; flex: 1; }
.sv-preflight-detail { color: var(--sv-text-2) !important; font-size: 0.74rem; }

/* ── FIX BLACK TEXT ON BLUE BUTTONS & PROGRESS BAR ─────────────────────── */
/* Forces all text inside buttons to inherit the white text color */
.stButton > button * {
    color: inherit !important;
}
[data-testid="stProgress"] * {
    color: inherit !important;
}

/* ── NATIVE RADIO CARDS (Perfect Grid & Hidden Circles) ────────────────── */
[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: stretch !important; 
    gap: 12px !important;
}

[data-testid="stRadio"] > div[role="radiogroup"] > label {
    flex: 1 1 0% !important; 
    min-width: 140px !important;
    background-color: var(--sv-surface) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: var(--sv-radius) !important;
    padding: 14px 16px !important;
    margin: 0 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* 1. HIDE THE NATIVE RADIO CIRCLE COMPLETELY */
[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* 2. CENTER THE TEXT INSIDE THE CARD */
[data-testid="stRadio"] > div[role="radiogroup"] > label > div:nth-child(2) {
    margin-left: 0 !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
}

[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    border-color: var(--sv-blue) !important;
    background-color: var(--sv-blue-dim) !important;
}

[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    border: 2px solid var(--sv-blue) !important;
    background-color: var(--sv-blue-dim) !important;
}

/* ── FIX SLIDER TEXT CONTRAST (Tuning Matrix) ──────────────────────────── */
/* Target the inactive text */
[data-testid="stTickBar"] div {
    color: var(--sv-text-2) !important;
}

/* Target the active text container to give it the blue bubble */
[data-testid="stTickBar"] div[style*="font-weight"] {
    background-color: var(--sv-blue) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
}

/* NUCLEAR OVERRIDE: Force the text inside the active bubble to be white */
[data-testid="stTickBar"] div[style*="font-weight"],
[data-testid="stTickBar"] div[style*="font-weight"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important; 
}

/* ── PROGRESS BAR TEXT FIX (Issue 5) ───────────────────────────────────── */
[data-testid="stProgress"] p { display: none !important; }
.sv-prog-label-text {
    text-align: center;
    font-size: 0.82rem;
    color: var(--sv-text-2) !important;
    margin: 4px 0 8px;
}

/* ── DARK MODE FALLBACK (Issue 5) ──────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --sv-bg:          #0f172a;
        --sv-surface:     #1e293b;
        --sv-surface-2:   #334155;
        --sv-border:      #334155;
        --sv-text:        #f1f5f9;
        --sv-text-2:      #94a3b8;
        --sv-text-3:      #64748b;
        --sv-blue-dim:    #1e3a5f;
    }
}        
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: step-complete banner (reused across tabs)
# ─────────────────────────────────────────────────────────────────────────────
 
def _complete_banner(message: str, next_tab: str) -> None:
    """Render a green completion banner with a next-step CTA label."""
    st.markdown(
        '<div class="sv-complete-banner">'
        + '<span class="sv-complete-text">✓ ' + message + '</span>'
        + '<span class="sv-complete-cta">' + next_tab + ' →</span>'
        + '</div>',
        unsafe_allow_html=True,
    )

def _locked_gate(required_steps: list[tuple[str, bool]]) -> None:
    """
    Render a locked-tab empty state.
 
    required_steps: list of (label, is_done) tuples describing what the user
                    still needs to complete before this tab unlocks.
                    e.g. [("Save Parameters (Tab 1)", True),
                           ("Generate Script (Tab 3)", False)]
    """
    first_todo = next((s[0] for s in required_steps if not s[1]), "previous step")
 
    steps_html = '<div class="sv-locked-steps">'
    for label, done in required_steps:
        icon_cls  = "sv-locked-step-done" if done else "sv-locked-step-todo"
        label_cls = "sv-locked-step-label-done" if done else "sv-locked-step-label-todo"
        icon_char = "✓" if done else "·"
        steps_html += (
            '<div class="sv-locked-step">'
            + f'<span class="sv-locked-step-icon {icon_cls}">{icon_char}</span>'
            + f'<span class="{label_cls}">{label}</span>'
            + '</div>'
        )
    steps_html += '</div>'
 
    st.markdown(
        '<div class="sv-locked-gate">'
        + '<span class="sv-locked-icon">🔒</span>'
        + f'<p class="sv-locked-title">Complete {first_todo} first</p>'
        + '<p class="sv-locked-body">This step will unlock once the required steps above are done. '
        + 'Use the tabs in order, or check the step tracker in the header.</p>'
        + steps_html
        + '</div>',
        unsafe_allow_html=True,
    )

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
TMP_ROOT          = tempfile.gettempdir()
UPLOAD_DIR        = os.path.join(TMP_ROOT, "sudovid_uploads")
KEYFRAME_DIR      = os.path.join(TMP_ROOT, "sudovid_keyframes")
PEXELS_CACHE_DIR  = os.path.join(TMP_ROOT, "sudovid_pexels_cache")
IMAGES_CACHE_DIR  = os.path.join(TMP_ROOT, "sudovid_images_cache")  # wiki/openverse stills
CARDS_DIR         = os.path.join(TMP_ROOT, "sudovid_cards")          # generated card PNGs

# Written by assembly worker; read by project-file generators
MANIFEST_FILE     = os.path.join(TMP_ROOT, "sudovid_clip_manifest.json")

for _d in (UPLOAD_DIR, KEYFRAME_DIR, PEXELS_CACHE_DIR, IMAGES_CACHE_DIR, CARDS_DIR):
    os.makedirs(_d, exist_ok=True)

# Model name — Gemini 3.5 Flash (stable, current flagship)
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
                      script_outline, script_text, is_shorts,
                      media_analysis):
    char_list  = ", ".join(character_names) if character_names else "none"
    outline    = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(script_outline))
    media_blk  = _media_analysis_block(media_analysis)

    if is_shorts:
        return f"""
You are a video editor cutting a YouTube SHORT (vertical 9:16, under 60 seconds).

FILM TITLE: {topic}
KNOWN CAST: {char_list}
SCRIPT: {script_text}

{media_blk}

Produce EXACTLY 3-6 segments, each 5-12 seconds.

Visual types for Shorts:
  "user_upload_video"  — user-provided video clip (HIGHEST PRIORITY if filename matches)
  "user_upload_image"  — user-provided image (HIGHEST PRIORITY if filename matches)
  "tmdb_poster"        — official film poster
  "tmdb_backdrop"      — cinematic scene still
  "text_card"          — punchy one-liner (max 1 total)

Return ONLY a valid JSON array. Schema per item:
{{
  "segment_index": 0, "type": "user_upload_video", "upload_filename": "",
  "duration_seconds": 8, "pexels_query": "", "cast_name": "",
  "text_line1": "", "text_line2": "", "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "", "wiki_title": "",
  "code_snippet": "", "code_language": "", "note": ""
}}
Rules:
- upload_filename: exact filename from USER UPLOADED MEDIA list, or "".
- Sum of duration_seconds must be 30-58 seconds total.
"""
    else:
        return f"""
You are a video editor planning a YouTube FILM / SERIES REVIEW video.

FILM TITLE: {topic}
SOURCE TYPE: {source_type}
TONE MATRIX: {matrix}
KNOWN CAST: {char_list}
SCRIPT OUTLINE:
{outline}

SCRIPT:
{script_text}

{media_blk}

Produce 8-20 segments (15-35s each). Visual types:
  "user_upload_video"  — user-provided video (HIGHEST PRIORITY)
  "user_upload_image"  — user-provided image (HIGHEST PRIORITY)
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
  "segment_index": 0, "type": "user_upload_video", "upload_filename": "",
  "duration_seconds": 20, "pexels_query": "", "cast_name": "",
  "text_line1": "", "text_line2": "", "stat_value": "", "stat_desc": "",
  "chapter_num": 0, "chapter_title": "", "wiki_title": "",
  "code_snippet": "", "code_language": "", "note": ""
}}
Rules:
- upload_filename: exact filename from USER UPLOADED MEDIA list, or "".
- pexels_query: 4-6 words SPECIFIC to this film's world (not genre labels).
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

def generate_shot_list(api_key: str, is_shorts: bool) -> list:
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
                                   script_outline, script_text, is_shorts, media_analysis)
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
# CLIP RESOLVER  — per-clip credit logic + path tracking for project export
# ─────────────────────────────────────────────────────────────────────────────
#
# Credit rules:
#   user_upload_*  → ""                        (creator's own content)
#   pexels_video   → "Video: Pexels.com"
#   pexels_image   → "Photo: Pexels.com"
#   trailer_clip   → "Courtesy: {topic} Official Trailer"
#   wiki/openverse → "Image: Wikipedia / Wikimedia Commons" / "Image: OpenVerse CC"
#   *_card         → ""                        (generated)
#
# Returns: (visual, credit_str, media_path, media_kind)
#   media_path  → absolute path on disk (for project file manifest)
#   media_kind  → "video" | "image"  (needed for NLE project XML)

def _save_image_to_cache(arr: np.ndarray, slug: str, subdir: str) -> str:
    """Save ndarray RGB image to subdir as PNG. Returns path."""
    os.makedirs(subdir, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)[:60]
    path = os.path.join(subdir, f"{safe}.png")
    Image.fromarray(arr.astype(np.uint8)).save(path, "PNG")
    return path


def _resolve_clip(shot, mode,
                  used_urls: set, topic,
                  cw: int, ch: int,
                  is_shorts: bool,
                  pexels_key: str,
                  media_analysis: list[dict]):
    """
    Resolve one shot dict → (visual, credit_str, media_path, media_kind).
    visual      → np.ndarray (still) or MoviePy clip
    credit_str  → per-source attribution or ""
    media_path  → absolute disk path (empty string if unavailable)
    media_kind  → "video" | "image"
    """
    stype        = shot.get("type", "text_card")
    upload_fn    = shot.get("upload_filename", "")
    arr          = None
    clip_credit  = ""
    media_path   = ""
    media_kind   = "image"   # default; overridden for video types

    # Build lookup: filename → path from media_analysis
    upload_map = {m["filename"]: m["path"] for m in media_analysis}

    def _fetch_unique(url):
        if not url or url in used_urls:
            return None
        a = _fetch_array(url, cw, ch)
        if a is not None:
            used_urls.add(url)
        return a

    # ── USER UPLOAD — VIDEO ──────────────────────────────────────────────────
    if stype == "user_upload_video":
        path = upload_map.get(upload_fn)
        if path and os.path.exists(path):
            try:
                duration   = max(float(shot.get("duration_seconds", 12)), 5.0)
                clip       = _loop_video_clip(path, duration, cw, ch)
                media_path = path
                media_kind = "video"
                return clip, "", media_path, media_kind
            except Exception:
                pass  # fall through to fallback

    # ── USER UPLOAD — IMAGE ──────────────────────────────────────────────────
    elif stype == "user_upload_image":
        path = upload_map.get(upload_fn)
        if path and os.path.exists(path):
            try:
                img        = Image.open(path).convert("RGB")
                arr        = _fit_to_canvas(img, cw, ch)
                media_path = path
                media_kind = "image"
                return arr, "", media_path, media_kind
            except Exception:
                pass

    # ── PEXELS VIDEO ─────────────────────────────────────────────────────────
    elif stype == "pexels_video":
        query       = shot.get("pexels_query") or topic
        orientation = "portrait" if is_shorts else "landscape"
        video_path  = _pexels_video_download(
            query, pexels_key, PEXELS_CACHE_DIR, orientation)
        if video_path:
            try:
                duration   = max(float(shot.get("duration_seconds", 12)), 5.0)
                clip       = _loop_video_clip(video_path, duration, cw, ch)
                media_path = video_path
                media_kind = "video"
                return clip, "Video: Pexels.com", media_path, media_kind
            except Exception:
                pass

    # ── PEXELS IMAGE ─────────────────────────────────────────────────────────
    elif stype == "pexels_image":
        query       = shot.get("pexels_query") or topic
        orientation = "portrait" if is_shorts else "landscape"
        url         = _pexels_image_url(query, pexels_key, orientation)
        arr         = _fetch_unique(url)
        if arr is not None:
            slug       = re.sub(r"[^a-z0-9]", "_", query.lower())[:40]
            media_path = _save_image_to_cache(arr, f"pexels_{slug}", PEXELS_CACHE_DIR)
            media_kind = "image"
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
            slug       = re.sub(r"[^a-z0-9]", "_", (queries[0] or topic).lower())[:40]
            media_path = _save_image_to_cache(arr, f"wiki_{slug}", IMAGES_CACHE_DIR)
            media_kind = "image"
            clip_credit = "Image: Wikipedia / Wikimedia Commons"

    # ── OPENVERSE / PEXELS_BROLL fallback ────────────────────────────────────
    elif stype in ("pexels_broll", "openverse_image"):
        query = (shot.get("pexels_query") or shot.get("image_query") or topic)
        if pexels_key:
            url = _pexels_image_url(query, pexels_key)
            arr = _fetch_unique(url)
            if arr is not None:
                slug       = re.sub(r"[^a-z0-9]", "_", query.lower())[:40]
                media_path = _save_image_to_cache(arr, f"pexels_{slug}", PEXELS_CACHE_DIR)
                media_kind = "image"
                clip_credit = "Photo: Pexels.com"
        if arr is None:
            for page in range(1, 4):
                url = _openverse_image(query, page=page)
                arr = _fetch_unique(url)
                if arr is not None:
                    slug       = re.sub(r"[^a-z0-9]", "_", query.lower())[:40]
                    media_path = _save_image_to_cache(arr, f"openverse_{slug}", IMAGES_CACHE_DIR)
                    media_kind = "image"
                    clip_credit = "Image: OpenVerse (CC licensed)"
                    break

    # ── GENERATED CARDS ──────────────────────────────────────────────────────
    elif stype == "text_card":
        arr = make_text_card(
            shot.get("text_line1", shot.get("note", topic)),
            shot.get("text_line2", ""), mode=mode, cw=cw, ch=ch)
        slug       = f"card_text_{shot.get('segment_index', 0):03d}"
        media_path = _save_image_to_cache(arr, slug, CARDS_DIR)
        media_kind = "image"

    elif stype == "stat_card":
        arr = make_stat_card(
            shot.get("stat_value", ""), shot.get("stat_desc", ""),
            mode=mode, cw=cw, ch=ch)
        slug       = f"card_stat_{shot.get('segment_index', 0):03d}"
        media_path = _save_image_to_cache(arr, slug, CARDS_DIR)
        media_kind = "image"

    elif stype == "code_card":
        arr = make_code_card(
            shot.get("code_snippet", "# No code provided"),
            shot.get("code_language", ""), cw=cw, ch=ch)
        slug       = f"card_code_{shot.get('segment_index', 0):03d}"
        media_path = _save_image_to_cache(arr, slug, CARDS_DIR)
        media_kind = "image"

    elif stype == "chapter_card":
        arr = make_chapter_card(
            int(shot.get("chapter_num", 1) or 1),
            shot.get("chapter_title", shot.get("note", "")),
            mode=mode, cw=cw, ch=ch)
        slug       = f"card_chapter_{shot.get('segment_index', 0):03d}"
        media_path = _save_image_to_cache(arr, slug, CARDS_DIR)
        media_kind = "image"

    # ── FALLBACK CHAIN ───────────────────────────────────────────────────────
    if arr is None and not hasattr(arr, "duration"):
        # 1. Pexels image on topic
        if pexels_key:
            url = _pexels_image_url(topic, pexels_key)
            arr = _fetch_unique(url)
            if arr is not None:
                slug       = re.sub(r"[^a-z0-9]", "_", topic.lower())[:40]
                media_path = _save_image_to_cache(arr, f"pexels_fb_{slug}", PEXELS_CACHE_DIR)
                media_kind = "image"
                clip_credit = "Photo: Pexels.com"
        # 2. OpenVerse
        if arr is None:
            url = _openverse_image(topic + " cinematic")
            arr = _fetch_unique(url)
            if arr is not None:
                slug       = re.sub(r"[^a-z0-9]", "_", topic.lower())[:40]
                media_path = _save_image_to_cache(arr, f"ov_fb_{slug}", IMAGES_CACHE_DIR)
                media_kind = "image"
                clip_credit = "Image: OpenVerse (CC licensed)"
        # 3. Wikipedia
        if arr is None:
            url = _wiki_image(topic)
            arr = _fetch_unique(url)
            if arr is not None:
                slug       = re.sub(r"[^a-z0-9]", "_", topic.lower())[:40]
                media_path = _save_image_to_cache(arr, f"wiki_fb_{slug}", IMAGES_CACHE_DIR)
                media_kind = "image"
                clip_credit = "Image: Wikipedia / Wikimedia Commons"
        # 4. Generated text card — always succeeds
        if arr is None:
            arr        = make_text_card(shot.get("note", topic), mode=mode, cw=cw, ch=ch)
            slug       = f"card_fallback_{shot.get('segment_index', 0):03d}"
            media_path = _save_image_to_cache(arr, slug, CARDS_DIR)
            media_kind = "image"
            clip_credit = ""

    return arr, clip_credit, media_path, media_kind
# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ASSEMBLY WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _assembly_worker(audio_path, shot_list, mode, topic,
                     output_path, is_shorts, credit_override, pexels_key,
                     media_analysis):
    try:
        cw, ch = _canvas(is_shorts)

        # ── Trailer (film mode only) ──────────────────────────────────────────
        if mode == MODE_FILM:
            _write_progress(5, "Film mode: External trailer scraping disabled for copyright safety.")

        # ── Build clips ───────────────────────────────────────────────────────
        total       = len(shot_list)
        used_urls   = set()
        clips       = []
        clip_credits = []
        clip_starts  = []
        running_time = 0.0

        manifest = []   # populated per clip; written to disk after render

        for i, shot in enumerate(shot_list):
            pct = 12 + int(68 * i / total)
            _write_progress(pct,
                f"Building clip {i+1}/{total} [{shot.get('type','?')}] — "
                f"{shot.get('note', shot.get('upload_filename', ''))}")

            duration = max(float(shot.get("duration_seconds", 12)), 5.0)
            zoom_in  = (i % 2 == 0)

            # SudoVid Update: Removed trailer_path, trailer_segs, trailer_idx
            visual, auto_credit, m_path, m_kind = _resolve_clip(
                shot, mode,
                used_urls, topic,
                cw, ch, is_shorts,
                pexels_key, media_analysis,
            )

            # visual is either an ndarray (still) or a MoviePy clip
            if hasattr(visual, "duration"):
                clip = visual.set_duration(min(visual.duration, duration))
            else:
                clip = _zoom_clip(visual, duration, zoom_in)

            clip_starts.append(running_time)
            clip_credits.append(auto_credit)
            manifest.append({
                "index":        i,
                "type":         shot.get("type", ""),
                "path":         m_path,
                "media_kind":   m_kind,
                "duration_sec": duration,
                "credit":       auto_credit,
                "label":        shot.get("note") or shot.get("upload_filename")
                                or shot.get("pexels_query") or shot.get("text_line1", ""),
            })
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

        # ── Write clip manifest for NLE project export ────────────────────────
        # Includes audio path so project generators can place the voiceover track
        manifest_data = {
            "clips":       manifest,
            "audio_path":  audio_path,
            "is_shorts":   is_shorts,
            "topic":       topic,
            "fps":         FPS,
            "canvas_w":    cw,
            "canvas_h":    ch,
        }
        try:
            with open(MANIFEST_FILE, "w") as mf:
                json.dump(manifest_data, mf, indent=2)
        except Exception:
            pass   # manifest failure must never block the render

        _write_progress(100, "Done!", done=True)

    except Exception as e:
        _write_progress(0, "", done=True, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# NLE PROJECT FILE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _frames(sec: float, fps: int = FPS) -> int:
    """Convert seconds to whole frame count."""
    return max(1, int(round(sec * fps)))


def _fcpxml_dur(sec: float, fps: int = FPS) -> str:
    """
    FCPXML rational time string for a duration in seconds.
    Uses 2400 timebase (standard for 24fps in FCPXML).
    e.g.  20s → "48000/2400s",  8.5s → "20400/2400s"
    """
    frames = _frames(sec, fps)
    return f"{frames * 100}/2400s"   # 2400 = 24fps * 100 sub-frame units


def generate_kdenlive_project(manifest_path: str, topic: str) -> bytes | None:
    """
    Build a KDEnlive 22+ project ZIP in memory.

    Structure:
        {topic}.kdenlive   — MLT XML referencing ./media/... (relative paths)
        media/
            voiceover.mp3
            000_clip.mp4 / 000_clip.png
            001_clip.jpg
            ...

    Returns ZIP bytes or None on failure.
    MLT XML targets KDEnlive 22+ (mlt version 7.x, kdenlive namespace).
    """
    import zipfile
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    try:
        with open(manifest_path) as f:
            mdata = json.load(f)
    except Exception:
        return None

    clips      = mdata["clips"]
    audio_path = mdata.get("audio_path", "")
    fps        = mdata.get("fps", FPS)
    cw         = mdata.get("canvas_w", 1280)
    ch         = mdata.get("canvas_h", 720)
    is_shorts  = mdata.get("is_shorts", False)

    safe_topic = re.sub(r"[^a-zA-Z0-9_-]", "_", topic)[:40]

    # ── Collect media files ───────────────────────────────────────────────────
    # media_items: list of { "arc_name": str, "src_path": str, "kind": str }
    media_items = []
    seen_paths  = {}   # src_path → arc_name (dedup sources used in multiple shots)

    def _add_media(src_path: str, kind: str, idx: int, label: str) -> str:
        """Register a media file; return its archive name inside media/."""
        if not src_path or not os.path.exists(src_path):
            return ""
        if src_path in seen_paths:
            return seen_paths[src_path]
        ext      = os.path.splitext(src_path)[1] or (".mp4" if kind == "video" else ".png")
        safe_lbl = re.sub(r"[^a-zA-Z0-9_-]", "_", label or "clip")[:30]
        arc_name = f"media/{idx:03d}_{safe_lbl}{ext}"
        media_items.append({"arc_name": arc_name, "src_path": src_path, "kind": kind})
        seen_paths[src_path] = arc_name
        return arc_name

    clip_arc_names = []
    for c in clips:
        arc = _add_media(c["path"], c["media_kind"], c["index"], c["label"])
        clip_arc_names.append(arc)

    # Voiceover
    audio_arc = ""
    if audio_path and os.path.exists(audio_path):
        ext       = os.path.splitext(audio_path)[1] or ".mp3"
        audio_arc = f"media/voiceover{ext}"
        media_items.append({"arc_name": audio_arc, "src_path": audio_path, "kind": "audio"})

    # ── Build MLT XML ─────────────────────────────────────────────────────────
    # KDEnlive 22+ wraps standard MLT XML with its own namespace attributes.
    mlt = ET.Element("mlt", {
        "LC_ALL":   "en_US.UTF-8",
        "version":  "7.22.0",
        "root":     ".",
        "producer": "main_bin",
    })

    # Profile
    ET.SubElement(mlt, "profile", {
        "description":    f"{'HD 1080p 24fps' if not is_shorts else 'Vertical 1080x1920 24fps'}",
        "width":          str(cw),
        "height":         str(ch),
        "progressive":    "1",
        "sample_aspect_num": "1",
        "sample_aspect_den": "1",
        "display_aspect_num": str(cw),
        "display_aspect_den": str(ch),
        "frame_rate_num": str(fps),
        "frame_rate_den": "1",
        "colorspace":     "709",
    })

    # ── Producers (one per unique media file) ─────────────────────────────────
    producer_ids = {}   # arc_name → producer id string

    # Black background producer (KDEnlive mandatory)
    black = ET.SubElement(mlt, "producer", {"id": "black_track", "in": "0", "out": "999999"})
    ET.SubElement(black, "property", {"name": "resource"}).text       = "black"
    ET.SubElement(black, "property", {"name": "mlt_service"}).text    = "color"
    ET.SubElement(black, "property", {"name": "kdenlive:clipname"}).text = "Black"

    for mi in media_items:
        if mi["kind"] == "audio":
            continue   # audio handled separately below
        pid  = f"producer_{len(producer_ids):04d}"
        kind = mi["kind"]
        arc  = mi["arc_name"]
        producer_ids[arc] = pid

        # Total frames for the out point — use a large sentinel for images
        if kind == "video":
            try:
                vc   = VideoFileClip(mi["src_path"])
                dur  = vc.duration
                vc.close()
            except Exception:
                dur = 30.0
            out_frame = max(0, _frames(dur, fps) - 1)
        else:
            out_frame = 999999   # images: KDEnlive treats as infinite

        prod = ET.SubElement(mlt, "producer", {
            "id":  pid,
            "in":  "0",
            "out": str(out_frame),
        })
        ET.SubElement(prod, "property", {"name": "resource"}).text     = f"./{arc}"
        ET.SubElement(prod, "property", {"name": "mlt_service"}).text  = (
            "avformat" if kind == "video" else "qimage"
        )
        ET.SubElement(prod, "property", {"name": "kdenlive:clipname"}).text = (
            os.path.basename(mi["src_path"])
        )
        ET.SubElement(prod, "property", {"name": "kdenlive:clip_type"}).text = (
            "1" if kind == "video" else "2"
        )

    # Audio producer for voiceover
    audio_pid = None
    if audio_arc:
        audio_pid = f"producer_{len(producer_ids):04d}"
        producer_ids[audio_arc] = audio_pid
        try:
            ac   = AudioFileClip(audio_path)
            adur = ac.duration
            ac.close()
        except Exception:
            adur = 120.0
        a_out = max(0, _frames(adur, fps) - 1)
        aprod = ET.SubElement(mlt, "producer", {
            "id":  audio_pid,
            "in":  "0",
            "out": str(a_out),
        })
        ET.SubElement(aprod, "property", {"name": "resource"}).text     = f"./{audio_arc}"
        ET.SubElement(aprod, "property", {"name": "mlt_service"}).text  = "avformat"
        ET.SubElement(aprod, "property", {"name": "kdenlive:clipname"}).text = "Voiceover"
        ET.SubElement(aprod, "property", {"name": "kdenlive:clip_type"}).text = "4"  # audio

    # ── Main tractor (timeline) ───────────────────────────────────────────────
    total_dur_sec = sum(c["duration_sec"] for c in clips)
    total_frames  = _frames(total_dur_sec, fps)

    tractor = ET.SubElement(mlt, "tractor", {
        "id":  "main_bin",
        "in":  "0",
        "out": str(total_frames - 1),
    })
    ET.SubElement(tractor, "property", {"name": "kdenlive:projectTractor"}).text = "1"

    # Track 0: black background (mandatory in KDEnlive)
    ET.SubElement(tractor, "track", {"producer": "black_track"})

    # Track 1: video/image playlist
    video_playlist_id = "playlist_video"
    ET.SubElement(tractor, "track", {"producer": video_playlist_id})

    # Track 2: audio (hidden video)
    if audio_pid:
        audio_playlist_id = "playlist_audio"
        ET.SubElement(tractor, "track", {
            "producer": audio_playlist_id,
            "hide":     "video",
        })

    # ── Video playlist ────────────────────────────────────────────────────────
    vpl = ET.SubElement(mlt, "playlist", {"id": video_playlist_id})
    current_frame = 0
    for c, arc in zip(clips, clip_arc_names):
        dur_frames = _frames(c["duration_sec"], fps)
        pid        = producer_ids.get(arc)
        if not pid:
            # No producer (file missing) — insert blank
            ET.SubElement(vpl, "blank", {"length": str(dur_frames)})
            current_frame += dur_frames
            continue
        kind = c["media_kind"]
        if kind == "video":
            out_f = min(dur_frames - 1, _frames(c["duration_sec"], fps) - 1)
        else:
            out_f = dur_frames - 1
        ET.SubElement(vpl, "entry", {
            "producer": pid,
            "in":       "0",
            "out":      str(out_f),
        })
        current_frame += dur_frames

    # ── Audio playlist ────────────────────────────────────────────────────────
    if audio_pid:
        apl = ET.SubElement(mlt, "playlist", {"id": audio_playlist_id})
        try:
            ac    = AudioFileClip(audio_path)
            adur  = ac.duration
            ac.close()
        except Exception:
            adur  = total_dur_sec
        ET.SubElement(apl, "entry", {
            "producer": audio_pid,
            "in":       "0",
            "out":      str(max(0, _frames(adur, fps) - 1)),
        })

    # ── Serialise XML ─────────────────────────────────────────────────────────
    raw_xml  = ET.tostring(mlt, encoding="unicode")
    pretty   = minidom.parseString(raw_xml).toprettyxml(indent="  ", encoding="utf-8")

    # ── Build ZIP in memory ───────────────────────────────────────────────────
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Project file
        zf.writestr(f"{safe_topic}.kdenlive", pretty)
        # Media files
        for mi in media_items:
            sp = mi["src_path"]
            if sp and os.path.exists(sp):
                zf.write(sp, mi["arc_name"])

    buf.seek(0)
    return buf.read()


def generate_fcpxml(manifest_path: str, topic: str) -> str | None:
    """
    Build an FCPXML v1.10 project string importable by:
      - DaVinci Resolve  (File → Import → Timeline)
      - Final Cut Pro    (native)
      - Premiere Pro     (via FCPXML import plugin)

    Media is referenced by absolute file:/// URIs — no bundling needed.
    User opens the .fcpxml in their NLE and re-links/auto-discovers media.

    Timecode uses 2400 sub-frame timebase (standard for 24fps FCPXML).
    """
    try:
        with open(manifest_path) as f:
            mdata = json.load(f)
    except Exception:
        return None

    clips      = mdata["clips"]
    audio_path = mdata.get("audio_path", "")
    fps        = mdata.get("fps", FPS)
    cw         = mdata.get("canvas_w", 1280)
    ch         = mdata.get("canvas_h", 720)

    safe_topic = re.sub(r"[^a-zA-Z0-9 _-]", "", topic)[:60] or "SudoVid Project"
    total_sec  = sum(c["duration_sec"] for c in clips)

    def _asset_id(idx):
        return f"a{idx + 1}"

    def _format_id():
        return "r1"

    def _file_uri(path: str) -> str:
        return "file://" + path.replace(" ", "%20")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE fcpxml>',
        '<fcpxml version="1.10">',
        '  <resources>',
        # Format resource
        f'    <format id="{_format_id()}" '
        f'name="FFVideoFormat{ch}p{fps}" '
        f'frameDuration="{100 * fps // fps}/2400s" '
        f'width="{cw}" height="{ch}"/>',
    ]

    # Asset per clip (deduplicate by path)
    seen_asset_paths = {}
    clip_asset_ids   = []

    for i, c in enumerate(clips):
        p = c.get("path", "")
        if p and os.path.exists(p):
            if p not in seen_asset_paths:
                aid  = _asset_id(len(seen_asset_paths))
                name = os.path.splitext(os.path.basename(p))[0]
                has_v = "1" if c["media_kind"] in ("video", "image") else "0"
                has_a = "1" if c["media_kind"] == "video" else "0"
                lines.append(
                    f'    <asset id="{aid}" name="{name}" '
                    f'src="{_file_uri(p)}" '
                    f'start="0s" duration="{_fcpxml_dur(c["duration_sec"], fps)}" '
                    f'hasVideo="{has_v}" hasAudio="{has_a}"/>'
                )
                seen_asset_paths[p] = aid
            clip_asset_ids.append(seen_asset_paths[p])
        else:
            clip_asset_ids.append(None)   # gap / generated card with no path

    # Voiceover asset
    audio_aid = None
    if audio_path and os.path.exists(audio_path):
        try:
            ac   = AudioFileClip(audio_path)
            adur = ac.duration
            ac.close()
        except Exception:
            adur = total_sec
        audio_aid = _asset_id(len(seen_asset_paths))
        lines.append(
            f'    <asset id="{audio_aid}" name="Voiceover" '
            f'src="{_file_uri(audio_path)}" '
            f'start="0s" duration="{_fcpxml_dur(adur, fps)}" '
            f'hasVideo="0" hasAudio="1"/>'
        )

    lines += [
        '  </resources>',
        '  <library>',
        f'    <event name="{safe_topic}">',
        f'      <project name="{safe_topic}">',
        f'        <sequence format="{_format_id()}" '
        f'duration="{_fcpxml_dur(total_sec, fps)}" '
        f'tcStart="0s" tcFormat="NDF" '
        f'audioLayout="stereo" audioRate="48k">',
        '          <spine>',
    ]

    # Video clips on spine
    offset_sec = 0.0
    for c, aid in zip(clips, clip_asset_ids):
        dur_sec = c["duration_sec"]
        name    = re.sub(r'[<>&"\']', "", c.get("label", "") or c["type"])
        if aid:
            lines.append(
                f'            <clip name="{name}" ref="{aid}" '
                f'offset="{_fcpxml_dur(offset_sec, fps)}" '
                f'duration="{_fcpxml_dur(dur_sec, fps)}" '
                f'start="0s"/>'
            )
        else:
            # Gap for clips with no resolvable file (shouldn't happen, but safe)
            lines.append(
                f'            <gap name="gap" '
                f'offset="{_fcpxml_dur(offset_sec, fps)}" '
                f'duration="{_fcpxml_dur(dur_sec, fps)}" '
                f'start="0s"/>'
            )
        offset_sec += dur_sec

    # Voiceover as connected audio clip anchored to first spine clip
    if audio_aid:
        try:
            ac   = AudioFileClip(audio_path)
            adur = ac.duration
            ac.close()
        except Exception:
            adur = total_sec
        lines += [
            f'            <audio name="Voiceover" ref="{audio_aid}" '
            f'lane="-1" '
            f'offset="0s" '
            f'duration="{_fcpxml_dur(adur, fps)}" '
            f'start="0s" role="dialogue"/>',
        ]

    lines += [
        '          </spine>',
        '        </sequence>',
        '      </project>',
        '    </event>',
        '  </library>',
        '</fcpxml>',
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_temp():
    """Remove ephemeral cache dirs and manifest. Called on Reset All Steps."""
    import shutil
    for d in (KEYFRAME_DIR, PEXELS_CACHE_DIR, IMAGES_CACHE_DIR, CARDS_DIR):
        try:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
    # Remove clip manifest so stale NLE exports are not offered
    try:
        os.remove(MANIFEST_FILE)
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION UI
# ─────────────────────────────────────────────────────────────────────────────

# ── HELPER: compute step states from session state ────────────────────────────
def _step_states() -> list[dict]:
    """
    Returns a list of 6 dicts, one per workflow step.
    Each dict has: label, short (sidebar), done (bool), active (bool).
    'active' = the earliest incomplete step.
    """
    ss = st.session_state
    steps = [
        {
            "label": "Parameters",
            "short": "Params",
            "done":  "topic_param" in ss,
        },
        {
            "label": "Research & Script",
            "short": "Script",
            "done":  "final_script_text" in ss,
        },
        {
            "label": "Voiceover",
            "short": "Voice",
            "done":  bool(ss.get("last_audio_path")),
        },
        {
            "label": "Video Assembly",
            "short": "Video",
            "done":  bool(ss.get("assembly_output")),
        },
        {
            "label": "Content Bundle",
            "short": "Bundle",
            "done":  "yt_bundle" in ss,
        },
    ]
    # Mark the first incomplete step as active
    found_active = False
    for s in steps:
        if not s["done"] and not found_active:
            s["active"] = True
            found_active = True
        else:
            s["active"] = False
    return steps
 
 
def _render_hero():
    """Render the branded hero header with inline step progress tracker."""
    steps = _step_states()
 
    # Build step dots HTML
    dots_html = ""
    for i, s in enumerate(steps):
        step_label = s["label"]
        if s["done"]:
            circle = (
                '<span class="sv-step-circle sv-step-done" '
                f'title="{step_label} — complete">✓</span>'
            )
        elif s.get("active"):
            circle = (
                '<span class="sv-step-circle sv-step-active" '
                f'title="{step_label} — current">{i + 1}</span>'
            )
        else:
            circle = (
                '<span class="sv-step-circle sv-step-locked" '
                f'title="{step_label} — locked">·</span>'
            )
 
        short = s["short"]
        connector = '<span class="sv-step-connector"></span>' if i < 5 else ""
        dots_html += (
            '<div class="sv-step-dot">'
            + circle
            + f'<span class="sv-step-label">{short}</span>'
            + "</div>"
            + connector
        )
 
    hero_html = (
        '<div class="sv-hero">'
        '<div class="sv-hero-left">'
        '<div class="sv-hero-icon">🎬</div>'
        '<div class="sv-hero-text">'
        '<span class="sv-hero-title">SudoVid</span>'
        '<span class="sv-hero-sub">AI-powered script, voice &amp; video engine</span>'
        "</div>"
        "</div>"
        '<div class="sv-step-track">'
        + dots_html
        + "</div>"
        "</div>"
    )
    st.markdown(hero_html, unsafe_allow_html=True)
 
 
_render_hero()

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
    # ── Branding strip ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sv-sidebar-brand">
            <span class="sv-sidebar-brand-icon">🎬</span>
            <span class="sv-sidebar-brand-name">SudoVid</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    # ── API Keys section ─────────────────────────────────────────────────────
    st.markdown(
        '<p class="sv-sidebar-section-label">API Keys</p>',
        unsafe_allow_html=True,
    )
 
    # Gemini
    if _gemini_from_secret:
        api_key = _secret("GEMINI_API_KEY")
        st.markdown(
            '<div class="sv-key-row sv-key-ok">'
            '<span class="sv-key-icon">✓</span>'
            '<span class="sv-key-name">Gemini</span>'
            '<span class="sv-key-source">from Secrets</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza…",
            help="Add GEMINI_API_KEY to Streamlit Secrets to hide this field",
            label_visibility="collapsed",
        )
        if api_key:
            st.markdown(
                '<div class="sv-key-row sv-key-ok">'
                '<span class="sv-key-icon">✓</span>'
                '<span class="sv-key-name">Gemini</span>'
                '<span class="sv-key-source">entered</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sv-key-row sv-key-missing">'
                '<span class="sv-key-icon">✗</span>'
                '<span class="sv-key-name">Gemini API key</span>'
                '<span class="sv-key-source">required</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.caption("Required for all AI steps.")
 
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
 
    # Pexels
    pexels_key = st.text_input(
        "Pexels API Key",
        type="password",
        placeholder="Pexels key…",
        value=_secret("PEXELS_API_KEY"),
        help=(
            "Required for Pexels images AND video B-roll. "
            "Free at pexels.com/api — or add PEXELS_API_KEY to Streamlit Secrets."
        ),
        label_visibility="collapsed",
    )
    if pexels_key:
        st.markdown(
            '<div class="sv-key-row sv-key-ok">'
            '<span class="sv-key-icon">✓</span>'
            '<span class="sv-key-name">Pexels</span>'
            '<span class="sv-key-source">images &amp; video</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sv-key-row sv-key-warn">'
            '<span class="sv-key-icon">⚠</span>'
            '<span class="sv-key-name">Pexels</span>'
            '<span class="sv-key-source">optional</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("Without Pexels, B-roll falls back to OpenVerse CC images.")
 
    st.divider()
 
    # ── Workflow progress (sidebar mini-tracker) ─────────────────────────────
    st.markdown(
        '<p class="sv-sidebar-section-label">Workflow progress</p>',
        unsafe_allow_html=True,
    )
    for i, s in enumerate(_step_states()):
        if s["done"]:
            icon, cls = "✓", "sv-prog-done"
        elif s.get("active"):
            icon, cls = "→", "sv-prog-active"
        else:
            icon, cls = "·", "sv-prog-locked"
        step_label = s["label"]
        st.markdown(
            f'<div class="sv-prog-row {cls}">'
            f'<span class="sv-prog-icon">{icon}</span>'
            f'<span class="sv-prog-label">{i + 1}. {step_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
 
    st.divider()
 
    # ── Reset — danger-styled with inline confirm ─────────────────────────────
    st.markdown(
        '<p class="sv-sidebar-section-label">Session</p>',
        unsafe_allow_html=True,
    )
 
    if st.session_state.get("_confirm_reset"):
        st.warning("This clears all steps and temp files. Continue?")
        col_y, col_n = st.columns(2)
        with col_y:
            if st.button("Yes, reset", key="btn_reset_confirm"):
                _cleanup_temp()
                st.session_state.clear()
                st.rerun()
        with col_n:
            if st.button("Cancel", key="btn_reset_cancel"):
                st.session_state["_confirm_reset"] = False
                st.rerun()
    else:
        if st.button("🗑 Reset All Steps", key="btn_reset_prompt"):
            st.session_state["_confirm_reset"] = True
            st.rerun()
 
    # ── Model info ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="sv-model-badge">Model: {GEMINI_MODEL}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS  (5 tabs — Media Upload merged into Video Assembly)
# ─────────────────────────────────────────────────────────────────────────────
(tab1, tab2, tab3, tab4, tab5) = st.tabs([
    "1. Parameters",
    "2. Research & Script",
    "3. Voiceover",
    "4. Video Assembly",
    "5. Content Bundle",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="sv-sidebar-section-label" style="margin-bottom:4px">Step 1 of 5</p>', unsafe_allow_html=True)
    st.subheader("Set Project Parameters & Angle")
    st.caption("Define the scope, tone, and your unique perspective before the AI conducts its research.")
    st.divider()
 
    topic = st.text_input("Topic or Title", placeholder="e.g. Project Hail Mary · CrowdStrike Outage")
    st.divider()
 
    # ── Content Mode ─────────────────────────────────────────────────────────
    st.markdown("**Content Mode**")
    st.caption("Selects the research persona and visual strategy.")
    
    mode_map = {
        "🎬 **Film & Series**": MODE_FILM,
        "🔍 **Tech News**": MODE_TECH,
        "📚 **Educational**": MODE_EDU
    }
    
    # Render native radio (CSS will transform this into the 3-column card grid)
    selected_mode_lbl = st.radio(
        "Content Mode",
        options=list(mode_map.keys()),
        captions=["Reviews, analysis", "Incidents, deep dives", "Tutorials, explainers"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Save the actual constant to state
    active_mode = mode_map[selected_mode_lbl]
    st.session_state["mode_param"] = active_mode
    
    st.divider()
 
    # ── Target Video Length ──────────────────────────────────────────────────
    st.markdown("**Target Video Length**")
    st.caption("Controls script length and output canvas format.")
    
    len_map = {
        "📱 **Shorts**": "Short-length (< 1 minute)",
        "🖥️ **Mid-length**": "Mid-length (3-8 mins)",
        "🖥️ **Deep Dive**": "Deep Dive (10+ mins)"
    }
    
    selected_len_lbl = st.radio(
        "Target Video Length",
        options=list(len_map.keys()),
        captions=["< 1 minute · 9:16", "3-8 mins · 16:9", "10+ mins · 16:9"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    video_length = len_map[selected_len_lbl]
    st.session_state["length_param"] = video_length
 
    source_type = "Original"
    if active_mode == MODE_FILM:
        st.divider()
        st.markdown("**Source Material**")
        source_type = st.radio("Source Material", ["Original", "Book", "Comic", "True Event", "Remake"], horizontal=True, label_visibility="collapsed")
 
    st.divider()
 
    st.markdown("**Tuning Matrix**")
    st.caption("Shape the script's tone and analytical lens.")
    st.markdown('<div class="sv-matrix">', unsafe_allow_html=True)
    matrix_data = {}
    if active_mode == MODE_FILM:
        c1, c2 = st.columns(2)
        with c1:
            matrix_data["Theory"] = st.select_slider("Film Theory Focus", ["Formalist", "Psychological", "Auteur", "Montage"])
            matrix_data["Visuals"] = st.select_slider("Visual Signature", ["Standard", "Stylized", "Iconic"])
        with c2:
            matrix_data["Fidelity"] = st.select_slider("Adaptation Fidelity", ["Loose", "Balanced", "Literal"])
            matrix_data["Tone"] = st.selectbox("Narrative Tone", ["Conversational", "Melancholic", "Frantic", "Academic"])
    elif active_mode == MODE_TECH:
        matrix_data["Severity"] = st.select_slider("Criticality", ["Bug", "Outage", "Crisis"])
        matrix_data["Scope"] = st.select_slider("User Impact", ["Niche", "Widespread", "Global"])
    else:
        matrix_data["Complexity"] = st.select_slider("Knowledge Level", ["Junior", "Senior", "Architect"])
        matrix_data["Method"] = st.select_slider("Pedagogical Style", ["Theory", "Mixed", "Practical"])
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.divider()
 
    st.markdown("**Your Unique Angle**")
    st.caption("The AI researches to support your perspective specifically — not to produce a generic summary.")
    angle_file = st.file_uploader("Upload rough draft (.txt)", type=["txt"])
    angle_text = st.text_area("Or type your angle here:", height=150)
 
    angle_len = len(angle_text)
    count_cls = "sv-char-count-warn" if angle_len > 1800 else ""
    st.markdown(f'<div class="sv-angle-footer"><span class="{count_cls}">{angle_len} / 2000 characters</span></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
 
    if st.button("💾 Save Parameters & Proceed", use_container_width=True):
        final_angle = angle_text.strip() or (angle_file.getvalue().decode("utf-8") if angle_file else "")
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
            _complete_banner("Parameters saved!", "Go to 2. Research & Script")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RESEARCH & SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="sv-sidebar-section-label" style="margin-bottom:4px">Step 2 of 5</p>', unsafe_allow_html=True)
    st.subheader("Research & Script")

    if "topic_param" not in st.session_state:
        _locked_gate([("Save Parameters (Tab 1)", False)])
    else:
        st.markdown("#### Phase 1 — Research")
        st.caption("Gemini searches the web for facts, dates, quotes, and data that specifically support your angle.")

        if st.button("🔍 Research Topic", key="btn_research"):
            if not api_key:
                st.warning("Gemini API Key required in sidebar.")
            else:
                with st.spinner(f"🌐 Researching **{st.session_state['topic_param']}**…"):
                    st.session_state["research"] = perform_grounded_research(
                        topic       = st.session_state["topic_param"],
                        mode        = st.session_state["mode_param"],
                        source_type = st.session_state["source_param"],
                        angle       = st.session_state["angle_param"],
                        length      = st.session_state["length_param"],
                        api_key     = api_key,
                    )
                st.session_state.pop("package", None)
                st.session_state.pop("final_script_text", None)

        if "research" in st.session_state:
            st.success("✅ Research complete")
            with st.expander("📄 View Factual Briefing", expanded=False):
                st.markdown(st.session_state["research"])
            st.markdown("---")
            
            st.markdown("#### Phase 2 — Generate Script")
            st.caption("Gemini synthesises the research and your angle into a full conversational script.")

            if st.button("📝 Generate Script", key="btn_generate_script"):
                with st.spinner(f"✍️ Writing script for **{st.session_state.get('length_param', '')}**…"):
                    st.session_state["package"] = generate_script_package(
                        mode           = st.session_state["mode_param"],
                        topic          = st.session_state["topic_param"],
                        research       = st.session_state["research"],
                        angle          = st.session_state["angle_param"],
                        matrix         = st.session_state["matrix_param"],
                        source_type    = st.session_state["source_param"],
                        length         = st.session_state["length_param"],
                        api_key        = api_key,
                        media_analysis = st.session_state.get("media_analysis", []),
                    )

            if "package" in st.session_state:
                p = st.session_state["package"]
                if "error" in p:
                    st.error(p["error"])
                else:
                    st.success(f"### {p.get('viral_title')}")
                    st.markdown("### 📝 Script Editor")
                    st.info("💡 Edit freely below — this is exactly what flows into Voiceover.")
 
                    fs = p.get("full_script", {})
                    default_text = "\n\n".join(filter(None, [
                        p.get("hook_script", ""), fs.get("intro", ""), fs.get("act1", ""),
                        fs.get("act2", ""), fs.get("act3", ""), fs.get("outro", ""),
                    ]))
 
                    st.session_state["final_script_text"] = st.text_area("Final Polish:", value=default_text.strip(), height=400)
 
                    script_body = st.session_state.get("final_script_text", "")
                    word_count  = len(script_body.split()) if script_body else 0
                    read_mins   = round(word_count / 130, 1) if word_count else 0
                    st.markdown(
                        f'<div class="sv-script-stats">'
                        f'<div class="sv-script-stat"><strong>{word_count:,}</strong> words</div>'
                        f'<div class="sv-script-stat">≈ <strong>{read_mins} min</strong> voiceover</div>'
                        f'</div>', unsafe_allow_html=True
                    )
 
                    st.download_button("📥 Download Script (.txt)", data=st.session_state["final_script_text"], file_name="script.txt", mime="text/plain")
                    _complete_banner("Script ready!", "Go to 3. Voiceover")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — VOICEOVER
# ─────────────────────────────────────────────────────────────────────────────
if "final_script_text" not in st.session_state:
        _locked_gate([("Save Parameters (Tab 1)",  "topic_param" in st.session_state), ("Generate Script (Tab 2)",  False)])
        st.stop()
        
VOICES = [
    ("en-US-ChristopherNeural", "Christopher", "Deep / Professional", "M"),
    ("en-US-GuyNeural",         "Guy",         "Natural / Conversational", "M"),
    ("en-US-EricNeural",        "Eric",        "Casual",              "M"),
    ("en-US-RogerNeural",       "Roger",       "Confident",           "M"),
    ("en-US-SteffanNeural",     "Steffan",     "Expressive",          "M"),
    ("en-US-AndrewNeural",      "Andrew",      "Warm",                "M"),
    ("en-US-BrianNeural",       "Brian",       "Crisp / News",        "M"),
    ("en-US-AriaNeural",        "Aria",        "Clear",               "F"),
    ("en-US-JennyNeural",       "Jenny",       "Conversational",      "F"),
    ("en-US-MichelleNeural",    "Michelle",    "Bright",              "F"),
    ("en-US-EmmaNeural",        "Emma",        "Friendly",            "F"),
    ("en-US-AvaNeural",         "Ava",         "Engaging",            "F"),
]
 
with tab3:
    st.markdown('<p class="sv-sidebar-section-label" style="margin-bottom:4px">Step 3 of 5</p>', unsafe_allow_html=True)
    st.subheader("AI Voiceover Studio")
    st.divider()
 
    st.markdown("**Select Narrator (US English)**")
    
    # Build options and captions dynamically from your VOICES list
    voice_options = []
    voice_captions = []
    voice_map = {}
    
    for v_id, v_name, v_style, v_gender in VOICES:
        emoji = "👨" if v_gender == "M" else "👩"
        lbl = f"{emoji} **{v_name}**"
        cap = f"{v_style} ({v_gender})"
        
        voice_options.append(lbl)
        voice_captions.append(cap)
        voice_map[lbl] = v_id  # Map visual label back to Edge TTS ID

    # Render native radio grid
    selected_voice_lbl = st.radio(
        "Select Narrator",
        options=voice_options,
        captions=voice_captions,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Save the selected ID to state
    st.session_state["voice_id"] = voice_map[selected_voice_lbl]
    
    # Get details for the synthesis preview
    selected_voice = next(v for v in VOICES if v[0] == st.session_state["voice_id"])
    voice_option = (selected_voice[0], f"{selected_voice[1]} ({selected_voice[3]})")
    st.divider()
 
    st.markdown("**Text Source**")
    source_mode = st.radio("Choose Text Source for Voiceover:", ["Use Generated Script", "Upload Custom Text File (.txt)"], label_visibility="collapsed")
 
    text_to_synthesize = st.session_state.get("final_script_text", "")
    if source_mode != "Use Generated Script":
        uploaded_file = st.file_uploader("Upload .txt for Voiceover", type=["txt"])
        if uploaded_file:
            text_to_synthesize = uploaded_file.getvalue().decode("utf-8")
 
    st.markdown("**Preview Text for Audio Generation**")
    
    # Just call st.text_area. The 'key' argument handles the state automatically.
    st.text_area(
        "This text will be sent to the AI Voice:", 
        value=text_to_synthesize, 
        height=250, 
        label_visibility="collapsed", 
        key="tab4_audio_text_input" 
    )
 
    if st.button("🔊 Generate Voiceover", use_container_width=True):
        audio_text = st.session_state.get("tab4_audio_text_input", "")
        
        if not audio_text.strip():
            st.error("Text box is empty.")
        else:
            with st.spinner(f"Synthesising with {voice_option[1]}…"):
                audio_path = generate_audio_sync(audio_text, voice_option[0])
                if audio_path:
                    st.session_state["last_audio_path"] = audio_path
                    st.session_state["audio_ready"] = True
                else:
                    st.error("Audio generation failed. Check internet connection.")

    # Render OUTSIDE button block for persistence
    if st.session_state.get("audio_ready") and os.path.exists(st.session_state.get("last_audio_path", "")):
        _ap = st.session_state["last_audio_path"]
        st.audio(_ap, format="audio/mp3")
        with open(_ap, "rb") as af:
            st.download_button("📥 Download Audio (.mp3)", data=af, file_name=f"voiceover.mp3", mime="audio/mp3", key="dl_audio_persist")
        _complete_banner("Voiceover ready!", "Go to 4. Video Assembly")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — VIDEO ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="sv-sidebar-section-label" style="margin-bottom:4px">Step 4 of 5</p>', unsafe_allow_html=True)
    st.subheader("Video Assembly")

    mode       = st.session_state.get("mode_param", "")
    topic_val  = st.session_state.get("topic_param", "")
    length_val = st.session_state.get("length_param", "")
    is_shorts  = "Short" in length_val

    # 1. Expandable Media Upload (Merged)
    with st.expander("📁 Upload Your Own Media — clips & images (optional)", expanded=False):
        st.warning("⚠️ **Copyright disclaimer:** Only upload media you own outright or have explicit rights to use. You are solely responsible for ensuring all uploaded clips and images are free of third-party copyright restrictions. SudoVid does not fetch copyrighted trailers.")
        
        col_v, col_i = st.columns(2)
        with col_v:
            uploaded_videos = st.file_uploader("Upload video clips", type=ALLOWED_VIDEO_TYPES, accept_multiple_files=True, key="video_up")
        with col_i:
            uploaded_images = st.file_uploader("Upload images", type=ALLOWED_IMAGE_TYPES, accept_multiple_files=True, key="image_up")
        
        col_save, col_skip = st.columns([3, 1])
        with col_save:
            if st.button("💾 Save & Analyse Media", use_container_width=True):
                saved = []
                for uf in (uploaded_videos or []):
                    if meta := _save_upload(uf, "video"): saved.append(meta)
                for uf in (uploaded_images or []):
                    if meta := _save_upload(uf, "image"): saved.append(meta)
                
                if saved:
                    st.session_state["uploaded_media"] = saved
                    with st.spinner(f"🔍 Analysing {len(saved)} file(s)..."):
                        st.session_state["media_analysis"] = analyse_uploaded_media(api_key, saved)
                    st.success(f"✅ {len(st.session_state['media_analysis'])} file(s) analysed.")
                else:
                    st.session_state["uploaded_media"] = []
                    st.session_state["media_analysis"] = []

        with col_skip:
            if st.button("Clear / Skip →", use_container_width=True):
                st.session_state["uploaded_media"] = []
                st.session_state["media_analysis"] = []
                st.success("Cleared.")

        analysis = st.session_state.get("media_analysis", [])
        if analysis:
            st.markdown("**Media Analysis Results**")
            for m in analysis:
                st.markdown(f"- **{m['filename']}** | {m.get('dominant_subjects', '—')}")

    override_on = st.checkbox("✏️ Override automatic credits with custom text")
    credit_override = st.text_input("Custom credit text", value=f"Courtesy: {topic_val}") if override_on else ""

    st.markdown("---")
    
    audio_path_v  = st.session_state.get("last_audio_path", "")
    script_text_v = st.session_state.get("final_script_text", "")
    all_ok = bool(audio_path_v) and bool(script_text_v) and bool(topic_val) and bool(api_key)
 
    if not all_ok:
        st.warning("⚠️ Pre-flight checks failed. Complete Tabs 1-3 first.")
        st.stop()

    if st.button("🎬 Generate Shot List"):
        with st.spinner("Gemini is planning your shot list…"):
            st.session_state["shot_list"] = generate_shot_list(api_key, is_shorts=is_shorts)

    if "shot_list" in st.session_state:
        sl = st.session_state["shot_list"]
        st.success(f"Shot list ready — **{len(sl)} clips**")

        with st.expander("📋 View Shot List", expanded=False):
            for s in sl:
                st.markdown(f"`{s.get('segment_index', 0):02d}` **{s.get('type','')}** — {s.get('duration_seconds','?')}s")

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
                        audio_path=audio_path_v, shot_list=sl, mode=mode,
                        topic=topic_val, output_path=output_path, is_shorts=is_shorts,
                        credit_override=credit_override, pexels_key=pexels_key,
                        media_analysis=st.session_state.get("media_analysis", []),
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
                st.session_state["assembly_done"] = True
                st.rerun()
            else:
                pct = prog.get("pct", 0)
                msg = prog.get("msg", "Working…")
                st.progress(pct / 100)
                st.markdown(f'<p class="sv-prog-label-text">{msg} ({pct}%)</p>', unsafe_allow_html=True)
                time.sleep(3)
                st.rerun()

    # Persistent Downloads (Outside the running block)
    if st.session_state.get("assembly_done"):
        out = st.session_state.get("assembly_output", os.path.join(TMP_ROOT, "sa_output_video.mp4"))
        if os.path.exists(out):
            st.success("✅ Video assembled successfully!")
            with open(out, "rb") as f:
                st.download_button(
                    "📥 Download MP4", data=f, 
                    file_name=f"{topic_val.replace(' ','_').lower()}_video.mp4", 
                    mime="video/mp4", key="dl_mp4_persist"
                )
            
            st.markdown("---")
            st.markdown("#### 🎞️ Export for Editing")
            manifest_exists = os.path.exists(MANIFEST_FILE)
            col_k, col_f = st.columns(2)
            
            with col_k:
                if manifest_exists:
                    if st.button("📦 Build KDEnlive Project (.zip)"):
                        with st.spinner("Packaging..."):
                            st.session_state["kdenlive_zip"] = generate_kdenlive_project(MANIFEST_FILE, topic_val)
                    if st.session_state.get("kdenlive_zip"):
                        st.download_button("⬇️ Download KDEnlive ZIP", data=st.session_state["kdenlive_zip"], file_name=f"{topic_val.replace(' ','_')}_kdenlive.zip", mime="application/zip", key="dl_kdenlive_persist")
            
            with col_f:
                if manifest_exists:
                    if st.button("🎬 Build FCPXML (.fcpxml)"):
                        with st.spinner("Generating..."):
                            st.session_state["fcpxml_str"] = generate_fcpxml(MANIFEST_FILE, topic_val)
                    if st.session_state.get("fcpxml_str"):
                        st.download_button("⬇️ Download FCPXML", data=st.session_state["fcpxml_str"].encode("utf-8"), file_name=f"{topic_val.replace(' ','_')}.fcpxml", mime="application/xml", key="dl_fcpxml_persist")
            
            if st.button("🔄 Reset & Assemble Again"):
                for k in ("shot_list", "assembly_running", "assembly_done", "kdenlive_zip", "fcpxml_str"):
                    st.session_state.pop(k, None)
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CONTENT BUNDLE  (optional post-production step)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="sv-sidebar-section-label" style="margin-bottom:4px">Step 5 of 5 — optional</p>', unsafe_allow_html=True)
    st.subheader("Content Bundle")
    st.caption("Generate your video title, description, tags, hashtags, and AI thumbnail prompt.")
    st.divider()
 
    bundle_source = st.radio("Base the bundle on:", ["Generated script (Tab 2)", "Final voiceover text (Tab 3)"])
 
    if st.button("📦 Generate Content Bundle", use_container_width=True):
        target_text = st.session_state.get("final_script_text", "") if "script" in bundle_source else st.session_state.get("tab4_audio_text", "")
        if not api_key: st.error("⚠️ Gemini API Key required.")
        elif not target_text.strip(): st.error("⚠️ No script text found.")
        else:
            with st.spinner("Generating content metadata…"):
                st.session_state["yt_bundle"] = generate_youtube_bundle(api_key, target_text)
 
    if "yt_bundle" in st.session_state:
        bundle = st.session_state["yt_bundle"]
        if "error" in bundle: st.error(bundle["error"])
        else:
            st.divider()
            st.text_input("Viral Title", value=bundle.get("viral_title", ""))
            st.text_area("Description", value=bundle.get("description", ""), height=160)
            col1, col2 = st.columns(2)
            with col1: st.text_area("Tags", value=", ".join(bundle.get("tags", [])), height=90)
            with col2: st.text_area("Hashtags", value=" ".join(bundle.get("hashtags", [])), height=90)
            st.text_area("🎨 AI Thumbnail Prompt", value=bundle.get("thumbnail_prompt", ""), height=120)
 
            _complete_banner("Content bundle ready!", "🎉 All steps complete! Your project is ready.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="sv-footer">
        <span>SudoVid v2.0</span>
        <span>Model: <code>{GEMINI_MODEL}</code></span>
        <span>AI-Powered Script, Voice &amp; Video Engine</span>
    </div>
    """,
    unsafe_allow_html=True,
)