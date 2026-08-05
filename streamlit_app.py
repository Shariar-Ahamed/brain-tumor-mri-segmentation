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

# 2. Seamless Full-Screen CSS Overrides (Removes Streamlit Padding, Margins & Double Scrollbars)
st.markdown("""
<style>
    /* Remove Streamlit Headers, Footers, and Sidebars */
    [data-testid="stHeader"], footer, [data-testid="stSidebar"], #MainMenu {
        display: none !important;
    }
    
    /* Remove All Outer Container Paddings and Margins */
    html, body, [data-testid="stAppViewContainer"], .main, .main .block-container {
        padding: 0px !important;
        margin: 0px !important;
        max-width: 100% !important;
        width: 100% !important;
        background-color: #0a0d14 !important;
    }

    /* Force iframe to expand 100% full width seamlessly without borders */
    .element-container, .stCustomComponentV1, iframe {
        width: 100% !important;
        border: none !important;
        margin: 0px !important;
        padding: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Helper to load and encode files
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

# 4. Read HTML, CSS, JS & Hero Image
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

# 5. Render Full-Screen UI (scrolling=False prevents double scrollbars)
st.components.v1.html(full_html, height=3200, scrolling=False)
