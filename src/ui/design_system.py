"""
Design System Module
====================
Centralized design tokens, themes, and reusable UI components
Following industry standards for consistent, accessible, and performant UIs
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime
import base64

# ============================================================================
# DESIGN TOKENS
# ============================================================================

class DesignTokens:
    """Central design tokens following design system principles"""
    
    # Color System - Modern Blue/Teal with excellent readability
    COLORS = {
        # Primary palette - Professional blue/teal
        'primary': '#0EA5E9',         # Sky blue - main brand color
        'primary_light': '#38BDF8',   # Lighter variant
        'primary_dark': '#0284C7',    # Darker variant
        'primary_alpha': 'rgba(14, 165, 233, 0.15)',  # Transparent variant for backgrounds
        
        # Neutral palette - Better contrast for readability
        'background': '#0F172A',      # Dark slate - main background
        'background_2': '#1E293B',    # Slate 800 - secondary background
        'surface': '#334155',         # Slate 700 - card/container background
        'surface_hover': '#475569',   # Slate 600 - hover state
        
        # Text colors - High contrast for readability
        'text_primary': '#F8FAFC',    # Almost white - main text
        'text_secondary': '#CBD5E1',  # Slate 300 - secondary text
        'text_disabled': '#94A3B8',   # Slate 400 - disabled text
        
        # Semantic colors - Vibrant and accessible
        'success': '#10B981',         # Emerald - success
        'warning': '#F59E0B',         # Amber - warnings
        'error': '#EF4444',           # Red - errors
        'info': '#06B6D4',            # Cyan - info
        
        # Accent colors for variety
        'accent_purple': '#A78BFA',   # Purple for special highlights
        'accent_pink': '#F472B6',     # Pink for notifications
        'accent_indigo': '#818CF8',   # Indigo for links
        
        # Border colors - Subtle but visible
        'border': 'rgba(148, 163, 184, 0.2)',      # Slate with transparency
        'border_hover': 'rgba(148, 163, 184, 0.3)', # Slightly more visible
        'border_focus': 'rgba(14, 165, 233, 0.5)',  # Primary color focus
    }
    
    # Typography System
    TYPOGRAPHY = {
        'font_family': 'Inter, system-ui, -apple-system, sans-serif',
        'font_sizes': {
            'xs': '0.75rem',    # 12px
            'sm': '0.875rem',   # 14px
            'base': '1rem',     # 16px
            'lg': '1.125rem',   # 18px
            'xl': '1.25rem',    # 20px
            '2xl': '1.5rem',    # 24px
            '3xl': '1.875rem',  # 30px
            '4xl': '2.25rem',   # 36px
        },
        'font_weights': {
            'normal': 400,
            'medium': 500,
            'semibold': 600,
            'bold': 700,
            'extrabold': 800,
        },
        'line_heights': {
            'tight': 1.25,
            'normal': 1.5,
            'relaxed': 1.75,
        }
    }
    
    # Spacing System (8px base grid)
    SPACING = {
        'xs': '0.25rem',   # 4px
        'sm': '0.5rem',    # 8px
        'md': '1rem',      # 16px
        'lg': '1.5rem',    # 24px
        'xl': '2rem',      # 32px
        '2xl': '3rem',     # 48px
        '3xl': '4rem',     # 64px
    }
    
    # Border Radius
    RADIUS = {
        'sm': '0.25rem',   # 4px
        'md': '0.5rem',    # 8px
        'lg': '0.75rem',   # 12px
        'xl': '1rem',      # 16px
        'full': '9999px',  # Fully rounded
    }
    
    # Shadows
    SHADOWS = {
        'sm': '0 2px 4px rgba(0,0,0,0.2)',
        'md': '0 4px 12px rgba(0,0,0,0.25)',
        'lg': '0 12px 28px rgba(0,0,0,0.30)',
        'xl': '0 20px 40px rgba(0,0,0,0.35)',
        'glow': '0 0 20px rgba(123, 17, 44, 0.3)',
    }
    
    # Transitions
    TRANSITIONS = {
        'fast': '120ms ease',
        'normal': '200ms ease',
        'slow': '300ms ease',
    }
    
    # Breakpoints for responsive design
    BREAKPOINTS = {
        'sm': 640,   # Mobile
        'md': 768,   # Tablet
        'lg': 1024,  # Desktop
        'xl': 1280,  # Wide desktop
    }


# ============================================================================
# CSS GENERATOR
# ============================================================================

def generate_css() -> str:
    """Generate comprehensive CSS based on design tokens"""
    tokens = DesignTokens()
    
    css = f"""
    <style>
    /* ========== CSS Variables ========== */
    :root {{
        /* Colors */
        --color-primary: {tokens.COLORS['primary']};
        --color-primary-light: {tokens.COLORS['primary_light']};
        --color-primary-dark: {tokens.COLORS['primary_dark']};
        --color-primary-alpha: {tokens.COLORS['primary_alpha']};
        
        --color-background: {tokens.COLORS['background']};
        --color-background-2: {tokens.COLORS['background_2']};
        --color-surface: {tokens.COLORS['surface']};
        --color-surface-hover: {tokens.COLORS['surface_hover']};
        
        --color-text-primary: {tokens.COLORS['text_primary']};
        --color-text-secondary: {tokens.COLORS['text_secondary']};
        --color-text-disabled: {tokens.COLORS['text_disabled']};
        
        --color-success: {tokens.COLORS['success']};
        --color-warning: {tokens.COLORS['warning']};
        --color-error: {tokens.COLORS['error']};
        --color-info: {tokens.COLORS['info']};
        
        --color-border: {tokens.COLORS['border']};
        --color-border-hover: {tokens.COLORS['border_hover']};
        --color-border-focus: {tokens.COLORS['border_focus']};
        
        /* Typography */
        --font-family: {tokens.TYPOGRAPHY['font_family']};
        
        /* Spacing */
        --spacing-xs: {tokens.SPACING['xs']};
        --spacing-sm: {tokens.SPACING['sm']};
        --spacing-md: {tokens.SPACING['md']};
        --spacing-lg: {tokens.SPACING['lg']};
        --spacing-xl: {tokens.SPACING['xl']};
        
        /* Radius */
        --radius-sm: {tokens.RADIUS['sm']};
        --radius-md: {tokens.RADIUS['md']};
        --radius-lg: {tokens.RADIUS['lg']};
        --radius-xl: {tokens.RADIUS['xl']};
        
        /* Shadows */
        --shadow-sm: {tokens.SHADOWS['sm']};
        --shadow-md: {tokens.SHADOWS['md']};
        --shadow-lg: {tokens.SHADOWS['lg']};
        --shadow-xl: {tokens.SHADOWS['xl']};
        --shadow-glow: {tokens.SHADOWS['glow']};
        
        /* Transitions */
        --transition-fast: {tokens.TRANSITIONS['fast']};
        --transition-normal: {tokens.TRANSITIONS['normal']};
        --transition-slow: {tokens.TRANSITIONS['slow']};
    }}
    
    /* ========== Global Styles ========== */
    * {{
        font-family: var(--font-family) !important;
    }}
    
    /* Improve overall readability */
    html, body {{
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }}
    
    /* Constrain content width for easier reading and add generous vertical rhythm */
    .block-container {{
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }}
    
    /* Typography for paragraphs and lists */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {{
        font-size: 1.05rem;
        line-height: 1.7;
        color: var(--color-text-primary);
    }}
    
    /* Links */
    [data-testid="stMarkdownContainer"] a {{
        color: var(--color-primary-light);
        text-decoration: none;
    }}
    [data-testid="stMarkdownContainer"] a:hover {{
        color: var(--color-primary);
        text-decoration: underline;
    }}
    
    /* Code blocks */
    pre, code, pre code {{
        background: var(--color-background-2) !important;
        color: var(--color-text-primary) !important;
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
    }}
    pre {{
        padding: 0.75rem 1rem;
        overflow: auto;
    }}
    
    /* Tables */
    .stTable table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .stTable th, .stTable td {{
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid var(--color-border);
    }}
    .stTable tr:nth-child(even) td {{
        background: rgba(255,255,255,0.02);
    }}
    
    /* Form labels and help text */
    label {{
        font-weight: 600 !important;
        color: var(--color-text-primary) !important;
    }}
    [data-testid="stMarkdownContainer"] small, .help-inline {{
        color: var(--color-text-secondary);
    }}
    
    /* Dividers spacing */
    hr, .stDividerHorizontal {{
        opacity: 0.5;
        margin: 1.25rem 0;
        border-color: var(--color-border);
    }}
    
    /* Subtle cleanup of default chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ opacity: 0.6; }}
    
    /* App Background with subtle gradient overlay */
    [data-testid="stAppViewContainer"] {{
        background: 
            radial-gradient(ellipse 1200px 700px at top right, 
                rgba(14, 165, 233, 0.03), transparent 40%),
            radial-gradient(ellipse 800px 600px at bottom left, 
                rgba(6, 182, 212, 0.02), transparent 40%),
            linear-gradient(180deg, var(--color-background), rgba(15, 23, 42, 0.95) 100%);
        color: var(--color-text-primary);
    }}
    
    /* Top accent bar */
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; 
        height: 3px;
        background: linear-gradient(90deg, 
            var(--color-primary), 
            var(--color-primary-light), 
            var(--color-primary));
        z-index: 9999;
    }}
    
    /* ========== Sidebar ========== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, 
            var(--color-background-2), 
            var(--color-background));
        border-right: 1px solid var(--color-border);
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: var(--color-text-primary);
    }}
    
    /* ========== Headers ========== */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--color-text-primary) !important;
        font-family: var(--font-family) !important;
    }}
    
    h1 {{
        font-size: {tokens.TYPOGRAPHY['font_sizes']['4xl']};
        font-weight: {tokens.TYPOGRAPHY['font_weights']['extrabold']};
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF, var(--color-primary-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: var(--spacing-lg);
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    h2 {{
        font-size: {tokens.TYPOGRAPHY['font_sizes']['2xl']};
        font-weight: {tokens.TYPOGRAPHY['font_weights']['bold']};
        border-left: 3px solid var(--color-primary-alpha);
        padding-left: var(--spacing-md);
        margin: var(--spacing-lg) 0 var(--spacing-md) 0;
    }}
    
    h3 {{
        font-size: {tokens.TYPOGRAPHY['font_sizes']['xl']};
        font-weight: {tokens.TYPOGRAPHY['font_weights']['semibold']};
        color: var(--color-text-primary);
        margin: var(--spacing-md) 0;
    }}
    
    /* ========== Buttons - Enhanced for Readability ========== */
    .stButton > button {{
        background: linear-gradient(135deg, 
            var(--color-primary), 
            var(--color-primary-light));
        border: 2px solid transparent;
        border-radius: var(--radius-lg);
        color: #FFFFFF;
        font-weight: {tokens.TYPOGRAPHY['font_weights']['semibold']};
        font-size: {tokens.TYPOGRAPHY['font_sizes']['base']};
        padding: 0.875rem 1.75rem;
        min-height: 48px;  /* Accessibility - minimum touch target */
        transition: all var(--transition-fast);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1), 
            0 2px 4px rgba(0,0,0,0.06),
            inset 0 1px 0 rgba(255,255,255,0.2);
        text-transform: none;
        letter-spacing: 0.025em;
        cursor: pointer;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        background: linear-gradient(135deg, 
            var(--color-primary-light), 
            var(--color-primary));
        box-shadow: 0 10px 15px rgba(0,0,0,0.15), 
            0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .stButton > button:focus {{
        outline: 3px solid var(--color-primary-alpha);
        outline-offset: 2px;
    }}
    
    /* Primary button - High contrast */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, 
            var(--color-primary), 
            var(--color-primary-dark));
        color: #FFFFFF;
        font-weight: {tokens.TYPOGRAPHY['font_weights']['bold']};
    }}
    
    /* Secondary button - Clear boundaries */
    .stButton > button[kind="secondary"] {{
        background: rgba(51, 65, 85, 0.5);
        border: 2px solid var(--color-border);
        color: var(--color-text-primary);
        backdrop-filter: blur(10px);
    }}
    
    .stButton > button[kind="secondary"]:hover {{
        background: rgba(71, 85, 105, 0.7);
        border-color: var(--color-primary);
        color: var(--color-primary-light);
    }}
    
    /* Button group spacing */
    .stButton {{
        margin: 0.25rem;
    }}
    
    /* Full width buttons on mobile */
    @media (max-width: 768px) {{
        .stButton > button {{
            width: 100%;
            padding: 1rem;
        }}
    }}
    
    /* ========== Forms & Inputs ========== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {{
        background: var(--color-surface) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--color-text-primary) !important;
        transition: all var(--transition-fast);
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 0 2px var(--color-primary-alpha) !important;
        outline: none !important;
    }}
    
    /* ========== File Uploader ========== */
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--color-surface);
        border: 2px dashed var(--color-primary-alpha);
        border-radius: var(--radius-lg);
        transition: all var(--transition-normal);
    }}
    
    [data-testid="stFileUploaderDropzone"]:hover {{
        background: var(--color-surface-hover);
        border-color: var(--color-primary);
        box-shadow: var(--shadow-md), var(--shadow-glow);
    }}
    
    /* ========== Alerts & Messages ========== */
    .stAlert {{
        background: var(--color-surface) !important;
        border-radius: var(--radius-lg) !important;
        border-left: 4px solid var(--color-primary) !important;
        box-shadow: var(--shadow-md);
    }}
    
    .stAlert[data-baseweb="notification"][kind="info"] {{
        border-left-color: var(--color-info) !important;
    }}
    
    .stAlert[data-baseweb="notification"][kind="success"] {{
        border-left-color: var(--color-success) !important;
    }}
    
    .stAlert[data-baseweb="notification"][kind="warning"] {{
        border-left-color: var(--color-warning) !important;
    }}
    
    .stAlert[data-baseweb="notification"][kind="error"] {{
        border-left-color: var(--color-error) !important;
    }}
    
    /* ========== Metrics ========== */
    [data-testid="metric-container"] {{
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--spacing-md);
        box-shadow: var(--shadow-sm);
        transition: all var(--transition-fast);
    }}
    
    [data-testid="metric-container"]:hover {{
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--color-primary-alpha);
    }}
    
    /* ========== Progress Bar ========== */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, 
            var(--color-primary), 
            var(--color-primary-light)) !important;
    }}
    
    /* ========== Expander ========== */
    .streamlit-expanderHeader {{
        background: var(--color-surface) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--color-text-primary) !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: var(--color-surface-hover) !important;
        border-color: var(--color-border-hover) !important;
    }}
    
    /* ========== Images ========== */
    [data-testid="stImage"] img {{
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
        box-shadow: var(--shadow-lg);
    }}
    
    /* ========== Custom Card Component ========== */
    .custom-card {{
        background: linear-gradient(145deg, 
            rgba(51, 65, 85, 0.6), 
            rgba(30, 41, 59, 0.8));
        backdrop-filter: blur(10px);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-xl);
        padding: var(--spacing-lg);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2),
                    0 1px 2px rgba(0,0,0,0.1),
                    inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all var(--transition-normal);
        position: relative;
        overflow: hidden;
    }}
    
    .custom-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        background: linear-gradient(90deg, 
            var(--color-primary), 
            var(--color-primary-light));
        opacity: 0;
        transition: opacity var(--transition-normal);
    }}
    
    .custom-card:hover {{
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl), var(--shadow-glow);
        border-color: var(--color-primary-alpha);
    }}
    
    .custom-card:hover::before {{
        opacity: 1;
    }}
    
    .custom-card-title {{
        font-size: {tokens.TYPOGRAPHY['font_sizes']['lg']};
        font-weight: {tokens.TYPOGRAPHY['font_weights']['bold']};
        color: var(--color-text-primary);
        margin-bottom: var(--spacing-sm);
    }}
    
    .custom-card-content {{
        color: var(--color-text-secondary);
        line-height: {tokens.TYPOGRAPHY['line_heights']['relaxed']};
    }}
    
    /* ========== Stat Card ========== */
    .stat-card {{
        background: var(--color-surface);
        border-radius: var(--radius-lg);
        padding: var(--spacing-md);
        border: 1px solid var(--color-border);
        transition: all var(--transition-fast);
        text-align: center;
    }}
    
    .stat-card:hover {{
        transform: scale(1.02);
        border-color: var(--color-primary-alpha);
        box-shadow: var(--shadow-md);
    }}
    
    .stat-value {{
        font-size: {tokens.TYPOGRAPHY['font_sizes']['3xl']};
        font-weight: {tokens.TYPOGRAPHY['font_weights']['bold']};
        background: linear-gradient(135deg, var(--color-primary), var(--color-primary-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .stat-label {{
        font-size: {tokens.TYPOGRAPHY['font_sizes']['sm']};
        color: var(--color-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* ========== Accessibility ========== */
    *:focus-visible {{
        outline: 2px solid var(--color-primary) !important;
        outline-offset: 2px !important;
    }}
    
    /* Skip to content link */
    .skip-link {{
        position: absolute;
        top: -40px;
        left: 0;
        background: var(--color-primary);
        color: white;
        padding: var(--spacing-sm) var(--spacing-md);
        text-decoration: none;
        border-radius: var(--radius-md);
        z-index: 100;
    }}
    
    .skip-link:focus {{
        top: var(--spacing-sm);
    }}
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {{
        :root {{
            --color-border: rgba(255,255,255,0.2);
            --color-text-secondary: #FFFFFF;
        }}
    }}
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {{
        * {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }}
    }}
    
    /* ========== Responsive Design ========== */
    @media (max-width: {tokens.BREAKPOINTS['md']}px) {{
        h1 {{
            font-size: {tokens.TYPOGRAPHY['font_sizes']['3xl']};
        }}
        
        h2 {{
            font-size: {tokens.TYPOGRAPHY['font_sizes']['xl']};
        }}
        
        .custom-card {{
            padding: var(--spacing-md);
        }}
    }}
    
    @media (max-width: {tokens.BREAKPOINTS['sm']}px) {{
        h1 {{
            font-size: {tokens.TYPOGRAPHY['font_sizes']['2xl']};
        }}
        
        .stButton > button {{
            width: 100%;
        }}
    }}
    </style>
    """
    return css


# ============================================================================
# REUSABLE UI COMPONENTS
# ============================================================================

def render_custom_card(
    title: str,
    content: str,
    icon: Optional[str] = None,
    color_accent: Optional[str] = None
) -> None:
    """Render a custom styled card component"""
    
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''
    accent_style = f'style="--color-primary: {color_accent};"' if color_accent else ''
    
    st.markdown(f"""
    <div class="custom-card" {accent_style}>
        <div class="custom-card-title">
            {icon_html}{title}
        </div>
        <div class="custom-card-content">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: Optional[str] = None
) -> None:
    """Render a statistics card"""
    
    delta_html = ''
    if delta:
        color = delta_color or DesignTokens.COLORS['success']
        delta_html = f'<div style="color: {color}; font-size: 0.875rem; margin-top: 0.25rem;">{delta}</div>'
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_progress_card(
    title: str,
    current: int,
    total: int,
    show_percentage: bool = True
) -> None:
    """Render a progress card with visual indicator"""
    
    percentage = (current / total * 100) if total > 0 else 0
    
    st.markdown(f"""
    <div class="custom-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600;">{title}</span>
            {f'<span>{percentage:.0f}%</span>' if show_percentage else ''}
        </div>
        <div style="background: var(--color-border); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light)); 
                        width: {percentage}%; height: 100%; transition: width 0.3s ease;">
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.875rem; color: var(--color-text-secondary);">
            <span>{current} completed</span>
            <span>{total} total</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_download_button(
    data: Dict[str, Any],
    filename: str,
    button_text: str = "Download",
    icon: str = "📥"
) -> None:
    """Create a styled download button for JSON data"""
    
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    
    st.markdown(f"""
    <a href="data:application/json;base64,{b64}" 
       download="{filename}"
       style="display: inline-flex; align-items: center; gap: 0.5rem;
              background: linear-gradient(135deg, var(--color-surface), var(--color-surface-hover));
              border: 1px solid var(--color-border);
              color: var(--color-text-primary);
              text-decoration: none;
              padding: 0.75rem 1.25rem;
              border-radius: var(--radius-lg);
              transition: all var(--transition-fast);
              font-weight: 500;">
        {icon} {button_text}
    </a>
    """, unsafe_allow_html=True)


def render_timeline_step(
    step_number: int,
    title: str,
    status: str = "pending",  # pending, active, completed
    description: Optional[str] = None
) -> None:
    """Render a timeline step indicator"""
    
    status_colors = {
        "pending": DesignTokens.COLORS['text_disabled'],
        "active": DesignTokens.COLORS['primary'],
        "completed": DesignTokens.COLORS['success']
    }
    
    status_icons = {
        "pending": "⏳",
        "active": "▶️",
        "completed": "✅"
    }
    
    color = status_colors.get(status, DesignTokens.COLORS['text_secondary'])
    icon = status_icons.get(status, "")
    
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
        <div style="position: relative;">
            <div style="width: 40px; height: 40px; 
                        background: {'var(--color-primary)' if status == 'active' else 'var(--color-surface)'};
                        border: 2px solid {color};
                        border-radius: 50%;
                        display: flex; align-items: center; justify-content: center;
                        font-weight: bold;
                        color: {'white' if status == 'active' else color};">
                {step_number}
            </div>
            <div style="position: absolute; top: 45px; left: 19px; bottom: -30px;
                        width: 2px; background: var(--color-border);"></div>
        </div>
        <div style="flex: 1;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <h4 style="margin: 0; color: {color};">{title}</h4>
                <span>{icon}</span>
            </div>
            {f'<p style="margin: 0.25rem 0; color: var(--color-text-secondary); font-size: 0.875rem;">{description}</p>' if description else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# LAYOUT HELPERS
# ============================================================================

@st.cache_data
def get_responsive_columns() -> Tuple[float, ...]:
    """Get responsive column ratios based on screen size"""
    # This is a simplified version - in production you'd use JavaScript
    # to detect actual screen size
    return (1, 2, 1)  # Sidebar, main, aside ratios


def render_breadcrumb(items: list) -> None:
    """Render a breadcrumb navigation"""
    
    breadcrumb_html = ' <span style="color: var(--color-text-disabled); margin: 0 0.5rem;">/</span> '.join(
        [f'<span style="color: var(--color-text-{"primary" if i == len(items)-1 else "secondary"});">{item}</span>' 
         for i, item in enumerate(items)]
    )
    
    st.markdown(f"""
    <div style="margin-bottom: 1rem; padding: 0.5rem; 
                background: var(--color-surface); 
                border-radius: var(--radius-md);
                font-size: 0.875rem;">
        {breadcrumb_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ACCESSIBILITY HELPERS
# ============================================================================

def add_aria_label(element_type: str, label: str) -> str:
    """Add ARIA labels for accessibility"""
    return f'aria-label="{label}" role="{element_type}"'


def render_skip_link() -> None:
    """Render skip to main content link for keyboard navigation"""
    st.markdown("""
    <a href="#main-content" class="skip-link">Skip to main content</a>
    """, unsafe_allow_html=True)


def set_focus_trap(container_id: str) -> None:
    """Set up focus trap for modal/dialog accessibility"""
    st.markdown(f"""
    <script>
    // Focus trap for accessibility
    document.addEventListener('DOMContentLoaded', function() {{
        const container = document.getElementById('{container_id}');
        if (container) {{
            const focusableElements = container.querySelectorAll(
                'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
            );
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];
            
            container.addEventListener('keydown', function(e) {{
                if (e.key === 'Tab') {{
                    if (e.shiftKey && document.activeElement === firstElement) {{
                        e.preventDefault();
                        lastElement.focus();
                    }} else if (!e.shiftKey && document.activeElement === lastElement) {{
                        e.preventDefault();
                        firstElement.focus();
                    }}
                }}
            }});
        }}
    }});
    </script>
    """, unsafe_allow_html=True)


# ============================================================================
# INITIALIZE DESIGN SYSTEM
# ============================================================================

def init_design_system() -> None:
    """Initialize the design system for the app"""
    # Inject CSS
    st.markdown(generate_css(), unsafe_allow_html=True)
    
    # Add accessibility features
    render_skip_link()
    
    # Add meta tags for better SEO and social sharing
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#7B112C">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
