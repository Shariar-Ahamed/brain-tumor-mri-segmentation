import os
import base64
import streamlit as st

# ==============================================================================
# NeuroSeg AI - Unified Explainable & Robust Brain Tumor MRI Segmentation App
# ==============================================================================

# 1. Page Configuration
st.set_page_config(
    page_title="NeuroSeg AI | Brain Tumor MRI Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Full-Screen CSS Overrides (Removes Streamlit default headers & double scrollbars)
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

# 3. Helper to load files and encode assets
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

# Load HTML, CSS, JS and Image assets
html_template = load_file_content("index.html")
css_content = load_file_content("styles.css")
js_content = load_file_content("app.js")
hero_b64 = get_base64_image("assets/brain_mri_hero.png")

if hero_b64 and html_template:
    html_template = html_template.replace("assets/brain_mri_hero.png", hero_b64)

# 4. Self-Contained Web Bundle
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

# 5. Render Web Application
st.components.v1.html(full_html_bundle, height=3200, scrolling=False)
