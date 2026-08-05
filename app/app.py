import os
import base64
import streamlit as st

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="NeuroSeg AI | Brain Tumor MRI Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit default header & padding to give full-screen HTML feel
st.markdown("""
<style>
    [data-testid="stHeader"] {
        display: none;
    }
    .main .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
    }
    iframe {
        border: none;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Helper to load and encode files
def load_file_content(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    return ""

# 3. Read HTML, CSS, JS & Hero Image
html_template = load_file_content("index.html")
css_content = load_file_content("styles.css")
js_content = load_file_content("app.js")
hero_b64 = get_base64_image("assets/brain_mri_hero.png")

# Replace relative image path with base64 data URI for Streamlit Cloud
if hero_b64:
    html_template = html_template.replace("assets/brain_mri_hero.png", hero_b64)

# Combine into self-contained HTML bundle
full_html = f"""
<!DOCTYPE html>
<html>
<head>
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

# 4. Render Exact HTML UI inside Streamlit
st.components.v1.html(full_html, height=2200, scrolling=True)
