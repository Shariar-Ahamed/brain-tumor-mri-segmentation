import os
import base64
import streamlit as st

# ==============================================================================
# NeuroSeg AI - Unified Explainable & Robust Brain Tumor MRI Segmentation App
# ==============================================================================

st.set_page_config(
    page_title="NeuroSeg AI | Brain Tumor MRI Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Full-Screen CSS Overrides for Seamless UI Rendering
st.markdown("""
<style>
    [data-testid="stHeader"], footer, [data-testid="stSidebar"], #MainMenu {
        display: none !important;
    }
    html, body, [data-testid="stAppViewContainer"], .main, .main .block-container {
        padding: 0px !important;
        margin: 0px !important;
        max-width: 100% !important;
        width: 100% !important;
        background-color: #0a0d14 !important;
    }
    .element-container, .stCustomComponentV1, iframe {
        width: 100% !important;
        border: none !important;
        margin: 0px !important;
        padding: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    return ""

def load_file_content(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Resolve paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(base_dir, "index.html")
css_path = os.path.join(base_dir, "styles.css")
js_path = os.path.join(base_dir, "app.js")
hero_path = os.path.join(base_dir, "assets", "brain_mri_hero.png")

html_template = load_file_content(html_path)
css_content = load_file_content(css_path)
js_content = load_file_content(js_path)
hero_b64 = get_base64_image(hero_path)

if hero_b64 and html_template:
    html_template = html_template.replace("assets/brain_mri_hero.png", hero_b64)

full_html_bundle = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
  {css_content}
  </style>
</head>
<body>
  {html_template}
  <script>
  {js_content}
  </script>
</body>
</html>
"""

st.components.v1.html(full_html_bundle, height=2800, scrolling=True)
