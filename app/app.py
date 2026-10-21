"""
app/app.py — Streamlit interactive demo for NQ Evidence Retrieval.

Tabs
----
1. Search        — free text or sample query, model selector, top-k slider,
                   result cards with highlighting, gold-relevance badges.
2. Evaluation    — performance table + PR curve + bar chart + statistical tests.
3. RRF Explainer — rank breakdown table for the last Hybrid query.

Theme support
-------------
- Dark / Light / System theme modes with segmented control switcher.
- 100% SVG vector icons for crisp, professional appearance.

Start with: streamlit run app/app.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, get_logger
from index import load_index
from rankers.tfidf import TFIDFRanker
from rankers.bm25 import BM25Ranker
from rankers.dense import DenseRanker
from rankers.hybrid import HybridRRFRanker

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SVG Icons Helper (Lucide / Minimalist 24x24 Vector System)
# ══════════════════════════════════════════════════════════════════════════════

def svg_icon(name: str, size: int = 16, color: str = "currentColor", extra_class: str = "") -> str:
    """Return an inline, crisp SVG icon string for HTML rendering."""
    cls = f"svg-ic {extra_class}".strip()

    if name == "search":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'
        )
    elif name == "chart":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line>'
            f'<line x1="6" y1="20" x2="6" y2="14"></line></svg>'
        )
    elif name == "lightning":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'
        )
    elif name == "brain":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-5.04z"></path>'
            f'<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-5.04z"></path></svg>'
        )
    elif name == "ruler":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M2 22h20"></path><path d="M22 22 2 2v20"></path><path d="M7 17l2-2"></path>'
            f'<path d="M12 12l2-2"></path><path d="M17 7l2-2"></path></svg>'
        )
    elif name == "compare":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<rect x="3" y="3" width="7" height="18" rx="1.5"></rect><rect x="14" y="3" width="7" height="18" rx="1.5"></rect></svg>'
        )
    elif name == "layers":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline>'
            f'<polyline points="2 12 12 17 22 12"></polyline></svg>'
        )
    elif name == "star":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}" '
            f'stroke="{color}" stroke-width="1.5">'
            f'<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>'
        )
    elif name == "check":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
        )
    elif name == "clock":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>'
        )
    elif name == "info":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line>'
            f'<line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
        )
    elif name == "alert":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>'
            f'<line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        )
    elif name == "database":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>'
            f'<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>'
        )
    elif name == "pin":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7z"></path>'
            f'<circle cx="12" cy="9" r="2.5"></circle></svg>'
        )
    elif name == "cpu":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect>'
            f'<line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line>'
            f'<line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line>'
            f'<line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line>'
            f'<line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>'
        )
    elif name == "sun":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line>'
            f'<line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>'
            f'<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line>'
            f'<line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>'
            f'<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
        )
    elif name == "moon":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>'
        )
    elif name == "monitor":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>'
            f'<line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>'
        )
    elif name == "logo":
        return (
            f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
            f'<circle cx="12" cy="12" r="9"></circle><polygon points="10 8 16 12 10 16 10 8" fill="{color}"></polygon></svg>'
        )
    return ""


MODEL_SVG_MAP = {
    "TF-IDF": "ruler",
    "BM25": "chart",
    "Dense": "brain",
    "Hybrid RRF": "lightning",
    "Compare All": "compare",
}

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NQ Evidence Retrieval",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Dynamic CSS Design System (Dark, Light, System themes)
# ══════════════════════════════════════════════════════════════════════════════

DARK_VARS = """
    --bg-app: #0e0e11;
    --bg-sidebar: #121216;
    --bg-card: #121216;
    --bg-card-subtle: #16161b;
    --border-card: #1c1c24;
    --border-hover: #2b4018;
    --border-input: #22222a;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #c8f043;
    --accent-bg: #1e2c12;
    --accent-border: #283a18;
    --gold-bg: #18140a;
    --gold-border: #3b2c0d;
    --gold-badge-bg: #2b1f09;
    --gold-badge-border: #d97706;
    --gold-badge-text: #fbd34d;
    --score-bg: #17171d;
    --score-border: #24242e;
    --score-text: #94a3b8;
    --rank-bg: #192710;
    --rank-border: #2d4a1a;
    --rank-text: #c8f043;
    --input-bg: #141418;
    --input-focus-border: #c8f043;
    --tab-active: #c8f043;
    --tab-inactive: #64748b;
    --header-badge-bg: #172110;
    --header-badge-border: #283a18;
    --header-badge-text: #c8f043;
    --query-pill-bg: #17240f;
    --query-pill-border: #294218;
    --query-pill-text: #c8f043;
    --mc-green-bg: linear-gradient(135deg, #15250f 0%, #101c0a 100%);
    --mc-green-border: #243d18;
    --mc-green-label: #86efac;
    --mc-blue-bg: linear-gradient(135deg, #0e1d35 0%, #091322 100%);
    --mc-blue-border: #183158;
    --mc-blue-label: #93c5fd;
    --mc-purple-bg: linear-gradient(135deg, #1b102b 0%, #12091d 100%);
    --mc-purple-border: #331d54;
    --mc-purple-label: #c4b5fd;
    --mc-teal-bg: linear-gradient(135deg, #092123 0%, #051416 100%);
    --mc-teal-border: #123336;
    --mc-teal-label: #5eead4;
    --info-bg: #0f1c30;
    --info-border: #1b3356;
    --info-text: #93c5fd;
    --warning-bg: #231908;
    --warning-border: #4a3410;
    --warning-text: #fcd34d;
    --code-bg: #111115;
    --mark-bg: rgba(200, 240, 67, 0.18);
    --mark-color: #c8f043;
    --col-header-bg: #15151a;
    --col-header-border: #22222a;
    --rrf-formula-bg: #15151a;
    --rrf-formula-border: #22222a;
"""

LIGHT_VARS = """
    --bg-app: #f8fafc;
    --bg-sidebar: #f1f5f9;
    --bg-card: #ffffff;
    --bg-card-subtle: #f8fafc;
    --border-card: #e2e8f0;
    --border-hover: #84cc16;
    --border-input: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --accent: #4d7c0f;
    --accent-bg: #f7fee7;
    --accent-border: #bef264;
    --gold-bg: #fffbeb;
    --gold-border: #fde68a;
    --gold-badge-bg: #fef3c7;
    --gold-badge-border: #f59e0b;
    --gold-badge-text: #92400e;
    --score-bg: #f1f5f9;
    --score-border: #e2e8f0;
    --score-text: #475569;
    --rank-bg: #f0fdf4;
    --rank-border: #bbf7d0;
    --rank-text: #166534;
    --input-bg: #ffffff;
    --input-focus-border: #65a30d;
    --tab-active: #3f6212;
    --tab-inactive: #64748b;
    --header-badge-bg: #f0fdf4;
    --header-badge-border: #bbf7d0;
    --header-badge-text: #166534;
    --query-pill-bg: #f0fdf4;
    --query-pill-border: #bbf7d0;
    --query-pill-text: #166534;
    --mc-green-bg: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    --mc-green-border: #86efac;
    --mc-green-label: #166534;
    --mc-blue-bg: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    --mc-blue-border: #93c5fd;
    --mc-blue-label: #1e40af;
    --mc-purple-bg: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    --mc-purple-border: #d8b4fe;
    --mc-purple-label: #6b21a8;
    --mc-teal-bg: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
    --mc-teal-border: #5eead4;
    --mc-teal-label: #115e59;
    --info-bg: #eff6ff;
    --info-border: #bfdbfe;
    --info-text: #1d4ed8;
    --warning-bg: #fffbeb;
    --warning-border: #fde68a;
    --warning-text: #92400e;
    --code-bg: #f8fafc;
    --mark-bg: rgba(132, 204, 22, 0.22);
    --mark-color: #365314;
    --col-header-bg: #ffffff;
    --col-header-border: #e2e8f0;
    --rrf-formula-bg: #ffffff;
    --rrf-formula-border: #e2e8f0;
"""

def inject_theme_css(theme_mode: str = "Dark"):
    """Inject CSS tokens based on selected theme mode (Dark, Light, or System)."""
    if theme_mode == "Dark":
        css_vars = f":root {{\n{DARK_VARS}\n}}"
    elif theme_mode == "Light":
        css_vars = f":root {{\n{LIGHT_VARS}\n}}"
    else:  # System
        css_vars = f":root {{\n{LIGHT_VARS}\n}}\n@media (prefers-color-scheme: dark) {{\n:root {{\n{DARK_VARS}\n}}\n}}"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    {css_vars}

    /* ── Base App Container ────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
        transition: background-color 0.2s ease, color 0.2s ease;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 1.8rem 2.4rem 2rem !important; max-width: 100% !important; }}

    .svg-ic {{
        display: inline-block;
        vertical-align: -0.18em;
        flex-shrink: 0;
    }}

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-card) !important;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }}
    [data-testid="stSidebar"] .block-container {{ padding: 1.6rem 1.1rem !important; }}

    .sidebar-logo {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 0 1.2rem 0;
        border-bottom: 1px solid var(--border-card);
        margin-bottom: 1.2rem;
    }}
    .sidebar-logo .icon-box {{
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #c8f043 0%, #9ae62e 100%);
        border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 12px rgba(200, 240, 67, 0.25);
    }}
    .sidebar-logo .name {{ font-size: 0.96rem; font-weight: 700; color: var(--text-primary); line-height: 1.2; letter-spacing: -0.01em; }}
    .sidebar-logo .sub  {{ font-size: 0.72rem; color: var(--text-muted); font-weight: 400; }}

    .nav-label {{
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
        color: var(--text-muted); text-transform: uppercase; margin: 1.1rem 0 0.5rem;
    }}

    .sidebar-meta {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        background: var(--bg-card-subtle);
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }}
    .sidebar-meta-item {{
        display: flex;
        align-items: center;
        gap: 9px;
    }}

    /* ── Segmented Control (Theme Switcher) ───────────────────────────── */
    [data-testid="stSegmentedControl"] {{
        width: 100% !important;
        margin-bottom: 0.5rem !important;
    }}
    [data-testid="stSegmentedControl"] div[data-baseweb="button-group"],
    [data-testid="stSegmentedControl"] div[role="radiogroup"] {{
        display: flex !important;
        width: 100% !important;
        gap: 0 !important;
    }}
    [data-testid="stSegmentedControl"] button {{
        flex: 1 1 0 !important;
        min-width: 0 !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-card) !important;
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.2rem !important;
        text-align: center !important;
        white-space: nowrap !important;
        transition: all 0.16s ease !important;
    }}
    [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
        background: var(--accent-bg) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        font-weight: 700 !important;
    }}

    /* ── Radio Buttons ────────────────────────────────────────────────── */
    [data-testid="stRadio"] > label {{ display: none !important; }}
    [data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 6px !important;
        flex-direction: column !important;
    }}
    [data-testid="stRadio"] label {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 10px !important;
        padding: 0.6rem 0.95rem !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        cursor: pointer !important;
        transition: all 0.16s ease-in-out !important;
    }}
    [data-testid="stRadio"] label:hover {{
        border-color: var(--border-hover) !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stRadio"] input[type="radio"]:checked + div {{
        background: var(--accent-bg) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        font-weight: 600 !important;
    }}

    /* ── Slider ───────────────────────────────────────────────────────── */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(200, 240, 67, 0.22) !important;
    }}
    [data-testid="stSlider"] [data-baseweb="track-foreground"] {{
        background: var(--accent) !important;
    }}

    /* ── Tabs ─────────────────────────────────────────────────────────── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 1px solid var(--border-card) !important;
        gap: 8px !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        background: transparent !important;
        border: none !important;
        color: var(--tab-inactive) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.3rem !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.18s ease-in-out !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{ color: var(--text-primary) !important; }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: var(--tab-active) !important;
        border-bottom-color: var(--tab-active) !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}

    /* ── Form Inputs ──────────────────────────────────────────────────── */
    [data-testid="stTextInput"] input {{
        background: var(--input-bg) !important;
        border: 1px solid var(--border-input) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1.1rem !important;
        font-size: 0.92rem !important;
    }}
    [data-testid="stTextInput"] input:focus {{
        border-color: var(--input-focus-border) !important;
        box-shadow: 0 0 0 3px rgba(200, 240, 67, 0.15) !important;
    }}
    [data-testid="stSelectbox"] > div > div {{
        background: var(--input-bg) !important;
        border: 1px solid var(--border-input) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }}

    /* ── Page Header ──────────────────────────────────────────────────── */
    .page-header {{ margin-bottom: 1.6rem; }}
    .header-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--header-badge-bg);
        border: 1px solid var(--header-badge-border);
        color: var(--header-badge-text);
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .page-header h1 {{
        font-size: 1.65rem; font-weight: 800;
        color: var(--text-primary); margin: 0 0 0.25rem;
        letter-spacing: -0.03em;
    }}
    .page-header p {{ font-size: 0.86rem; color: var(--text-muted); margin: 0; }}

    /* ── Metric Card Grid ─────────────────────────────────────────────── */
    .metric-card {{
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .metric-card.green  {{ background: var(--mc-green-bg); border: 1px solid var(--mc-green-border); }}
    .metric-card.blue   {{ background: var(--mc-blue-bg); border: 1px solid var(--mc-blue-border); }}
    .metric-card.purple {{ background: var(--mc-purple-bg); border: 1px solid var(--mc-purple-border); }}
    .metric-card.teal   {{ background: var(--mc-teal-bg); border: 1px solid var(--mc-teal-border); }}

    .mc-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.45rem;
    }}
    .mc-label {{
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .metric-card.green  .mc-label {{ color: var(--mc-green-label); }}
    .metric-card.blue   .mc-label {{ color: var(--mc-blue-label); }}
    .metric-card.purple .mc-label {{ color: var(--mc-purple-label); }}
    .metric-card.teal   .mc-label {{ color: var(--mc-teal-label); }}

    .mc-value {{
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.04em;
        line-height: 1.1;
    }}
    .mc-sub {{ font-size: 0.72rem; color: var(--text-muted); margin-top: 0.35rem; }}

    /* ── Query Pill ───────────────────────────────────────────────────── */
    .query-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--query-pill-bg);
        border: 1px solid var(--query-pill-border);
        border-radius: 999px;
        padding: 0.35rem 0.95rem;
        font-size: 0.82rem;
        color: var(--query-pill-text);
        font-weight: 500;
        margin-bottom: 1.1rem;
    }}

    /* ── Result Card ──────────────────────────────────────────────────── */
    .result-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 14px;
        padding: 1.05rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.16s ease, border-color 0.16s ease;
    }}
    .result-card:hover {{
        border-color: var(--border-hover);
        transform: translateY(-1px);
    }}
    .result-card.gold {{
        border-color: var(--gold-border);
        background: var(--gold-bg);
    }}

    .rc-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.65rem;
        flex-wrap: wrap;
    }}
    .rank-badge {{
        width: 26px; height: 26px;
        background: var(--rank-bg);
        border: 1px solid var(--rank-border);
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--rank-text);
        flex-shrink: 0;
    }}
    .rank-badge.gold-rank {{
        background: var(--gold-badge-bg);
        border-color: var(--gold-badge-border);
        color: var(--gold-badge-text);
    }}
    .doc-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        flex: 1; min-width: 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .gold-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: var(--gold-badge-bg);
        border: 1px solid var(--gold-badge-border);
        color: var(--gold-badge-text);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    .score-badge {{
        background: var(--score-bg);
        border: 1px solid var(--score-border);
        color: var(--score-text);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.74rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        margin-left: auto;
    }}
    .passage-text {{
        font-size: 0.84rem;
        color: var(--text-secondary);
        line-height: 1.65;
    }}
    mark {{
        background: var(--mark-bg);
        color: var(--mark-color);
        padding: 1px 4px;
        border-radius: 4px;
        font-weight: 600;
    }}

    /* ── Compare Headers ──────────────────────────────────────────────── */
    .col-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: var(--col-header-bg);
        border: 1px solid var(--col-header-border);
        border-radius: 11px;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.85rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-primary);
    }}
    .col-header .ms {{ color: var(--text-muted); font-weight: 400; font-size: 0.74rem; }}

    /* ── Section Cards ────────────────────────────────────────────────── */
    .section-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 16px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.3rem;
    }}
    .section-card h3 {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.94rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 1rem;
    }}

    /* ── Info & Warning Cards ─────────────────────────────────────────── */
    .info-card {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background: var(--info-bg);
        border: 1px solid var(--info-border);
        border-radius: 13px;
        padding: 1rem 1.25rem;
        font-size: 0.85rem;
        color: var(--info-text);
        line-height: 1.55;
        margin-bottom: 1.1rem;
    }}
    .info-card.warning {{
        background: var(--warning-bg);
        border-color: var(--warning-border);
        color: var(--warning-text);
    }}
    .info-icon {{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }}

    /* ── Eval Image Wrapper ───────────────────────────────────────────── */
    .eval-img-wrap {{
        background: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 15px;
        padding: 1.2rem;
        margin-bottom: 1.1rem;
    }}
    .eval-img-wrap h4 {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 0.85rem;
    }}

    /* ── RRF Formula ──────────────────────────────────────────────────── */
    .rrf-formula {{
        background: var(--rrf-formula-bg);
        border: 1px solid var(--rrf-formula-border);
        border-left: 3px solid var(--accent);
        border-radius: 0 12px 12px 0;
        padding: 0.85rem 1.15rem;
        margin: 1.1rem 0;
    }}
    .formula-title {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--accent);
        margin-bottom: 0.35rem;
    }}
    .rrf-formula code {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: var(--text-primary);
    }}

    .divider-label {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 1.4rem 0 1rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
    }}
    .divider-label::before, .divider-label::after {{
        content: ''; flex: 1; height: 1px; background: var(--border-card);
    }}

    /* ── Dataframe & Code ─────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid var(--border-card) !important;
    }}
    [data-testid="stCode"] {{
        background: var(--code-bg) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 12px !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading indexes and neural models …")
def load_all(config_path: str = "config.yaml"):
    """Load config, index, and all rankers once at startup."""
    cfg = load_config(ROOT / config_path)
    data_dir = ROOT / cfg["dataset"]["data_dir"]

    corpus_df = pd.read_parquet(data_dir / "corpus.parquet")
    query_df  = pd.read_parquet(data_dir / "queries.parquet")
    qrels_df  = pd.read_parquet(data_dir / "qrels.parquet")

    corpus_dict: Dict[str, Dict] = {
        row["id"]: {"title": row.get("title", ""), "text": row["text"]}
        for _, row in corpus_df.iterrows()
    }
    qrels_dict: Dict[str, Dict[str, int]] = {}
    for row in qrels_df.itertuples():
        qrels_dict.setdefault(row.query_id, {})[row.corpus_id] = int(row.score)

    sample_queries = query_df.sample(
        n=min(cfg["app"]["sample_queries_count"], len(query_df)), random_state=42
    )[["id", "text"]].to_dict("records")

    index = load_index(ROOT / cfg["index"]["index_path"])

    tfidf_ranker = TFIDFRanker(index,
                               tf_scheme=cfg["tfidf"]["tf_scheme"],
                               normalize=cfg["tfidf"]["normalize"])
    bm25_ranker = BM25Ranker(index, k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])

    dense_cfg = cfg["dense"]
    dense_ranker = DenseRanker(
        model_name=dense_cfg["model_name"],
        embeddings_path=str(ROOT / dense_cfg["embeddings_path"]),
        doc_ids_path=str(ROOT / dense_cfg["doc_ids_path"]),
        batch_size=dense_cfg["batch_size"],
        normalize_embeddings=dense_cfg["normalize_embeddings"],
        use_faiss=dense_cfg["use_faiss"],
        max_seq_length=cfg["preprocessing"]["max_transformer_tokens"],
    )
    dense_ranker.load()

    rrf_cfg = cfg["rrf"]
    hybrid_ranker = HybridRRFRanker(
        rankers={"tfidf": tfidf_ranker, "bm25": bm25_ranker, "dense": dense_ranker},
        k=rrf_cfg["k"],
        candidates_per_method=rrf_cfg["candidates_per_method"],
    )

    rankers = {
        "TF-IDF":     tfidf_ranker,
        "BM25":       bm25_ranker,
        "Dense":      dense_ranker,
        "Hybrid RRF": hybrid_ranker,
    }

    return cfg, rankers, corpus_dict, qrels_dict, sample_queries


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def highlight_query_terms(text: str, query: str) -> str:
    tokens = [t.strip() for t in query.lower().split() if len(t.strip()) > 2]
    for token in set(tokens):
        text = re.sub(rf"(?i)\b({re.escape(token)})\b", r"<mark>\1</mark>", text)
    return text


def render_result_card(rank: int, doc_id: str, score: float,
                       corpus_dict: Dict, query: str,
                       is_gold: bool, max_chars: int = 380) -> None:
    doc   = corpus_dict.get(doc_id, {"title": doc_id, "text": "(passage not found)"})
    title = doc.get("title", "") or doc_id
    text  = doc.get("text", "")
    snippet = text[:max_chars] + ("…" if len(text) > max_chars else "")
    highlighted = highlight_query_terms(snippet, query)

    rank_cls = "gold-rank" if is_gold else ""
    card_cls = "gold"      if is_gold else ""
    gold_html = f'<span class="gold-badge">{svg_icon("star", 11, "#f59e0b")} Relevant</span>' if is_gold else ""

    card_html = (
        f'<div class="result-card {card_cls}">'
        f'<div class="rc-header">'
        f'<span class="rank-badge {rank_cls}">{rank}</span>'
        f'<span class="doc-title" title="{title}">{title}</span>'
        f'{gold_html}'
        f'<span class="score-badge">{score:.4f}</span>'
        f'</div>'
        f'<div class="passage-text">{highlighted}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def run_search(ranker_name: str, rankers: Dict, query: str, k: int):
    ranker = rankers[ranker_name]
    t0 = time.perf_counter()
    if ranker_name == "Hybrid RRF":
        results, breakdown = ranker.search_with_breakdown(query, k=k)
        st.session_state["last_rrf_breakdown"] = breakdown
        st.session_state["last_rrf_query"]     = query
    else:
        results = ranker.search(query, k=k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(cfg):
    with st.sidebar:
        st.markdown(
            f'<div class="sidebar-logo">'
            f'<div class="icon-box">{svg_icon("lightning", 20, "#0e0e11")}</div>'
            f'<div>'
            f'<div class="name">NQ Evidence Engine</div>'
            f'<div class="sub">Open-Domain QA Retrieval</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-label">Theme Mode</div>', unsafe_allow_html=True)
        theme_options = ["Dark", "Light", "System"]
        selected_theme = st.segmented_control(
            "Theme Mode",
            options=theme_options,
            selection_mode="single",
            required=True,
            width="stretch",
            key="theme_mode",
            label_visibility="collapsed",
        ) or st.session_state.get("theme_mode", "Dark")

        st.markdown('<div class="nav-label">Retrieval Paradigm</div>', unsafe_allow_html=True)
        model_choice = st.radio(
            "model",
            ["TF-IDF", "BM25", "Dense", "Hybrid RRF", "Compare All"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="nav-label">Results per Query</div>', unsafe_allow_html=True)
        k = st.slider(
            "Top-K",
            min_value=cfg["app"]["min_top_k"],
            max_value=cfg["app"]["max_top_k"],
            value=cfg["app"]["default_top_k"],
            label_visibility="collapsed",
        )
        st.caption(f"Displaying top **{k}** ranked passages")

        st.markdown('<div class="nav-label">Corpus & Benchmark</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-meta">'
            f'<div class="sidebar-meta-item">{svg_icon("database", 14, "#818cf8")} <span>100,000 Wikipedia passages</span></div>'
            f'<div class="sidebar-meta-item">{svg_icon("search", 14, "#34d399")} <span>1,000 NQ test queries</span></div>'
            f'<div class="sidebar-meta-item">{svg_icon("pin", 14, "#f472b6")} <span>Seed 42 · BEIR benchmark</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    return model_choice, k, selected_theme


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Search
# ══════════════════════════════════════════════════════════════════════════════

def tab_search(cfg, rankers, corpus_dict, qrels_dict, sample_queries, model_choice, k):
    st.markdown(
        f'<div class="page-header">'
        f'<div class="header-badge">{svg_icon("search", 11, "#c8f043")} Retrieval Playground</div>'
        f'<h1>Evidence Search</h1>'
        f'<p>Retrieve supporting Wikipedia evidence passages for real Natural Questions</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_q, col_s = st.columns([5, 2])
    with col_q:
        query = st.text_input(
            "Question",
            placeholder="Type your question (e.g. who sang the national anthem at super bowl 2016) …",
            key="search_query",
            label_visibility="collapsed",
        )
    with col_s:
        sample_labels = ["— Pick a sample query —"] + [
            (q["text"][:55] + "…" if len(q["text"]) > 55 else q["text"])
            for q in sample_queries
        ]
        chosen = st.selectbox("Samples", sample_labels, key="sample_select",
                              label_visibility="collapsed")
        if chosen != "— Pick a sample query —":
            query = chosen

    if not query.strip():
        st.markdown(
            f'<div class="info-card">'
            f'<div class="info-icon">{svg_icon("info", 18, "#60a5fa")}</div>'
            f'<div class="info-text">Type a natural question above or select a sample query to retrieve supporting evidence. '
            f'Switch models in the sidebar to compare TF-IDF, BM25, Dense embeddings, and Hybrid RRF.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Gold relevance check
    qid_lookup = {q["text"]: q["id"] for q in sample_queries}
    query_id   = qid_lookup.get(query)
    gold_ids   = set()
    if query_id and query_id in qrels_dict:
        gold_ids = {d for d, s in qrels_dict[query_id].items() if s > 0}

    if model_choice == "Compare All":
        st.markdown(
            f'<div class="query-pill">{svg_icon("search", 13, "#c8f043")} <span>{query}</span></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        for col, mname in zip(cols, ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]):
            with col:
                results, ms = run_search(mname, rankers, query, k)
                col_icon = svg_icon(MODEL_SVG_MAP.get(mname, "cpu"), 14, "#c8f043")
                st.markdown(
                    f'<div class="col-header">{col_icon} <span>{mname}</span><span class="ms">· {ms:.0f} ms</span></div>',
                    unsafe_allow_html=True,
                )
                for rank, (doc_id, score) in enumerate(results, 1):
                    render_result_card(rank, doc_id, score, corpus_dict, query,
                                       is_gold=(doc_id in gold_ids))
    else:
        results, ms = run_search(model_choice, rankers, query, k)
        hits = sum(1 for d, _ in results if d in gold_ids)

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(
            f'<div class="metric-card green">'
            f'<div class="mc-top"><span class="mc-label">Passages Found</span>{svg_icon("layers", 16, "#86efac")}</div>'
            f'<div class="mc-value">{len(results)}</div>'
            f'<div class="mc-sub">top-{k} retrieved</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="metric-card blue">'
            f'<div class="mc-top"><span class="mc-label">Gold Hits</span>{svg_icon("check", 16, "#93c5fd")}</div>'
            f'<div class="mc-value">{hits}</div>'
            f'<div class="mc-sub">ground-truth matches</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="metric-card purple">'
            f'<div class="mc-top"><span class="mc-label">Latency</span>{svg_icon("clock", 16, "#c4b5fd")}</div>'
            f'<div class="mc-value">{ms:.0f}<span style="font-size:1.1rem;font-weight:400;color:var(--text-muted)"> ms</span></div>'
            f'<div class="mc-sub">query execution time</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        m4.markdown(
            f'<div class="metric-card teal">'
            f'<div class="mc-top"><span class="mc-label">Active Model</span>{svg_icon(MODEL_SVG_MAP.get(model_choice, "cpu"), 16, "#5eead4")}</div>'
            f'<div class="mc-value" style="font-size:1.35rem">{model_choice}</div>'
            f'<div class="mc-sub">ranking paradigm</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="query-pill">{svg_icon("search", 13, "#c8f043")} <span>{query}</span></div>',
            unsafe_allow_html=True,
        )
        for rank, (doc_id, score) in enumerate(results, 1):
            render_result_card(rank, doc_id, score, corpus_dict, query,
                               is_gold=(doc_id in gold_ids))


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def tab_evaluation(theme_mode: str = "Dark"):
    st.markdown(
        f'<div class="page-header">'
        f'<div class="header-badge">{svg_icon("chart", 11, "#60a5fa")} Benchmark Analysis</div>'
        f'<h1>Evaluation Results</h1>'
        f'<p>Comparative performance across 800 test queries on BEIR Natural Questions</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    results_dir = ROOT / "results"

    table_path = results_dir / "performance_table.csv"
    if table_path.exists():
        df = pd.read_csv(table_path, index_col=0)
        highlight_color = "#dcfce7" if theme_mode == "Light" else "#1e3a18"
        highlight_text  = "color:#166534;font-weight:700" if theme_mode == "Light" else "color:#c8f043;font-weight:700"
        styled = df.style.format("{:.4f}").highlight_max(
            axis=0, color=highlight_color, props=highlight_text
        )
        st.markdown(
            f'<div class="section-card"><h3>{svg_icon("chart", 16, "#c8f043")} Performance Benchmark Table</h3>',
            unsafe_allow_html=True,
        )
        st.dataframe(styled, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="info-card warning">'
            f'<div class="info-icon">{svg_icon("alert", 18, "#fbbf24")}</div>'
            f'<div class="info-text">No evaluation results found. Run <code>python src/run_experiments.py</code> to generate benchmarks.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    chart_pairs = [
        ("pr_curve.png",              "Precision-Recall Curves (11-point interpolated)", "chart"),
        ("metrics_bar.png",           "Grouped Metrics Comparison Bar Chart",           "chart"),
        ("ndcg_histogram.png",        "nDCG@10 Score Distribution Across Queries",       "layers"),
        ("bm25_vs_dense_winloss.png", "BM25 vs Dense Win/Loss Differential Analysis",    "compare"),
    ]
    col_l, col_r = st.columns(2)
    col_cycle = [col_l, col_r, col_l, col_r]
    for (fname, label, icon_name), col in zip(chart_pairs, col_cycle):
        p = results_dir / fname
        if p.exists():
            with col:
                st.markdown(
                    f'<div class="eval-img-wrap">'
                    f'<h4>{svg_icon(icon_name, 14, "#94a3b8")} {label}</h4>',
                    unsafe_allow_html=True,
                )
                st.image(str(p), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    sig_path = results_dir / "significance.txt"
    if sig_path.exists():
        st.markdown(
            f'<div class="section-card"><h3>{svg_icon("brain", 16, "#c8f043")} Statistical Significance (Paired t-test & Wilcoxon)</h3>',
            unsafe_allow_html=True,
        )
        st.code(sig_path.read_text(encoding="utf-8"), language="text")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — RRF Explainer
# ══════════════════════════════════════════════════════════════════════════════

def tab_rrf_explainer(theme_mode: str = "Dark"):
    st.markdown(
        f'<div class="page-header">'
        f'<div class="header-badge">{svg_icon("lightning", 11, "#c8f043")} Fusion Math</div>'
        f'<h1>RRF Rank Fusion Explainer</h1>'
        f'<p>Step-by-step breakdown of Reciprocal Rank Fusion combining TF-IDF, BM25, and Dense ranks</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    breakdown = st.session_state.get("last_rrf_breakdown")
    query     = st.session_state.get("last_rrf_query", "")

    if not breakdown:
        st.markdown(
            f'<div class="info-card">'
            f'<div class="info-icon">{svg_icon("info", 18, "#60a5fa")}</div>'
            f'<div class="info-text">Execute a <strong>Hybrid RRF</strong> search in the Search tab first, '
            f'then return here to inspect the rank fusion breakdown for that specific query.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="query-pill">{svg_icon("search", 13, "#c8f043")} <span>{query}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="rrf-formula">'
        f'<div class="formula-title">{svg_icon("lightning", 13, "#c8f043")} Mathematical Formulation (k=60)</div>'
        f'<code>RRF_score(d) = &Sigma;<sub>m &isin; M</sub> 1 / (60 + rank<sub>m</sub>(d))</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for entry in breakdown:
        row = {
            "Document": (entry["doc_id"][:24] + "…" if len(entry["doc_id"]) > 24 else entry["doc_id"]),
            "RRF Score": round(entry["rrf_score"], 5),
        }
        for method, rank in entry["ranks"].items():
            row[f"{method} Rank"]    = str(rank) if rank is not None else "—"
            row[f"{method} 1/(k+r)"] = (
                round(entry["contributions"][method], 5)
                if entry["contributions"][method] > 0 else 0.0
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    rrf_cols   = [c for c in df.columns if "1/(k+r)" in c]
    fmt_dict   = {"RRF Score": "{:.5f}", **{c: "{:.5f}" for c in rrf_cols}}

    highlight_color = "#dcfce7" if theme_mode == "Light" else "#1e3a18"
    highlight_text  = "color:#166534;font-weight:700" if theme_mode == "Light" else "color:#c8f043;font-weight:700"

    st.dataframe(
        df.style.format(fmt_dict).highlight_max(
            subset=["RRF Score"], color=highlight_color,
            props=highlight_text
        ),
        use_container_width=True,
    )

    # Per-method contribution bar chart (top 10)
    st.markdown('<div class="divider-label"><span>Per-Method Contribution (Top 10 Documents)</span></div>',
                unsafe_allow_html=True)
    methods = list(breakdown[0]["ranks"].keys()) if breakdown else []
    if methods:
        contrib_data = {
            m: [e["contributions"].get(m, 0) for e in breakdown[:10]]
            for m in methods
        }
        contrib_df = pd.DataFrame(
            contrib_data,
            index=[e["doc_id"][:16] + "…" for e in breakdown[:10]],
        )
        st.bar_chart(contrib_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    current_theme = st.session_state.get("theme_mode", "Dark")
    inject_theme_css(current_theme)

    try:
        cfg, rankers, corpus_dict, qrels_dict, sample_queries = load_all()
    except FileNotFoundError as e:
        st.markdown(
            f'<div style="max-width:600px;margin:4rem auto;text-align:center">'
            f'<div style="margin-bottom:1rem">{svg_icon("alert", 48, "#fbbf24")}</div>'
            f'<h2 style="color:var(--text-primary);margin:0 0 0.5rem">Setup Incomplete</h2>'
            f'<p style="color:var(--text-muted);margin-bottom:1.5rem">{e}</p>'
            f'<div style="background:var(--bg-card);border:1px solid var(--border-card);border-radius:12px;padding:1rem;text-align:left">'
            f'<code style="color:var(--accent)">python run_all.py</code>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    model_choice, k, active_theme = render_sidebar(cfg)

    tab1, tab2, tab3 = st.tabs(["Search", "Evaluation", "RRF Explainer"])
    with tab1:
        tab_search(cfg, rankers, corpus_dict, qrels_dict, sample_queries, model_choice, k)
    with tab2:
        tab_evaluation(active_theme)
    with tab3:
        tab_rrf_explainer(active_theme)


if __name__ == "__main__":
    main()
