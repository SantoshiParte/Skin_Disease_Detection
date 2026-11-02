import streamlit as st
import pandas as pd
import hashlib
import re
import html as html_lib

from PIL import Image
from pathlib import Path
from datetime import datetime
from io import BytesIO

from utils.predictions import SkinDiseasePredictor
from utils.disease_info import get_disease_info


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkinSense AI | Skin Image Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
MAX_IMAGE_SIZE_MB = 10
APP_VERSION = "1.0.0"


# ============================================================
# 2. HTML RENDER HELPER  (fixes the "raw HTML shown as code" bug)
# ============================================================
# Markdown treats lines indented by 4+ spaces as a code block, and a
# blank line ends an HTML block. So we strip indentation and blank
# lines before sending HTML to st.markdown.

def render_html(markup: str):
    cleaned = "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value) -> str:
    """Escape dynamic text before inserting it into HTML."""
    return html_lib.escape(str(value))


# ============================================================
# 3. CUSTOM CSS
# ============================================================

render_html("""
<style>
.stApp { background: #F4F7FB; }
.block-container { max-width: 1350px; padding-top: 2rem; padding-bottom: 3rem; }

/* Readable text colours in the main area for Light, Dark and System themes.
   :where() keeps specificity at zero so component styles below always win. */
:where(.block-container) :where(h1, h2, h3, h4, h5, h6) { color: #17324D !important; }
:where(.block-container) :where(p, li, label, strong, em, td, th, summary) { color: #3F566B !important; }
:where(.block-container) :where([data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *) { color: #64748B !important; }
:where(.block-container) :where([data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span) { color: #3F566B !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #102A43; }
section[data-testid="stSidebar"] * { color: #FFFFFF; }
section[data-testid="stSidebar"] .stRadio label { padding: 9px 10px; border-radius: 9px; }
section[data-testid="stSidebar"] .stRadio label:hover { background: #1D4562; }

/* Hero */
.hero { background: linear-gradient(120deg, #102A43, #176B87); padding: 35px; border-radius: 18px; margin-bottom: 25px; box-shadow: 0 8px 25px rgba(16,42,67,0.12); }
.hero h1 { color: #FFFFFF !important; font-size: 34px; font-weight: 750; margin-bottom: 8px; }
.hero p { color: #D8EDF5 !important; font-size: 16px; margin-bottom: 0; }

/* Cards */
.info-card { background: #FFFFFF; border: 1px solid #E3EAF2; border-radius: 15px; padding: 23px; min-height: 175px; box-shadow: 0 3px 12px rgba(20,50,75,0.04); }
.info-card h3 { color: #176B87 !important; font-size: 19px; margin-bottom: 12px; }
.info-card p { color: #64748B !important; font-size: 14px; line-height: 1.7; }

.list-card { background: #FFFFFF; border: 1px solid #E3EAF2; border-radius: 15px; padding: 22px 24px; min-height: 150px; box-shadow: 0 3px 12px rgba(20,50,75,0.04); margin-bottom: 14px; }
.list-card h3 { color: #176B87 !important; font-size: 19px; margin: 0 0 12px 0; }
.list-card ul { margin: 0; padding-left: 20px; }
.list-card li { color: #475B6F !important; font-size: 14.5px; line-height: 1.75; margin-bottom: 5px; }
.list-card .empty { color: #64748B !important; font-size: 14px; }

/* Metrics */
.metric-card { background: #FFFFFF; border: 1px solid #E3EAF2; border-radius: 14px; padding: 20px; min-height: 115px; }
.metric-label { color: #64748B; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
.metric-value { color: #176B87; font-size: 26px; font-weight: 750; margin-top: 8px; }

/* Result */
.result-card { background: #FFFFFF; border: 1px solid #DCE6EF; border-radius: 16px; padding: 24px; margin-bottom: 15px; min-height: 130px; }
.result-title { color: #176B87; font-size: 24px; font-weight: 750; margin-top: 8px; line-height: 1.3; }
.result-caption { color: #64748B; font-size: 13px; margin-top: 6px; }
.section-label { color: #176B87; font-size: 12px; font-weight: 750; letter-spacing: 1.4px; text-transform: uppercase; }

/* Upload */
[data-testid="stFileUploader"] { background: #FFFFFF; border: 1px dashed #70A8B9; border-radius: 14px; padding: 18px; }
[data-testid="stFileUploaderDropzone"] { background: #F4F7FB !important; }
[data-testid="stFileUploaderDropzone"] button { background: #FFFFFF !important; border: 1px solid #B8CAD8 !important; }
[data-testid="stFileUploaderDropzone"] button * { color: #17324D !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius: 10px; min-height: 43px; font-weight: 600; transition: 0.2s ease; }
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"], .stDownloadButton > button { background: #FFFFFF; border: 1px solid #B8CAD8; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { background: #176B87; border: none; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { background: #12546C; }
.stButton > button[kind="primary"] *, .stButton > button[data-testid="stBaseButton-primary"] * { color: #FFFFFF !important; }

/* Tabs */
button[role="tab"][aria-selected="true"] p { color: #176B87 !important; font-weight: 700; }
[data-baseweb="tab-highlight"] { background: #176B87 !important; }

/* Footer */
.footer { text-align: center; border-top: 1px solid #DCE6EF; margin-top: 45px; padding: 25px 10px; color: #718096; font-size: 13px; line-height: 1.9; }
.footer strong { color: #176B87 !important; }

div[data-testid="stAlert"] { border-radius: 10px; }
hr { border-color: #E2E8F0; }
</style>
""")


# ============================================================
# 4. SESSION STATE
# ============================================================

st.session_state.setdefault("history", [])
st.session_state.setdefault("last_prediction", None)
st.session_state.setdefault("navigation", "Home")


def go_to_analysis():
    st.session_state.navigation = "Skin Analysis"


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path, modified_time):
    return SkinDiseasePredictor(model_path)


# ============================================================
# 6. COMMON UI FUNCTIONS
# ============================================================

def display_header(title, subtitle):
    render_html(f"""
    <div class="hero">
        <h1>{esc(title)}</h1>
        <p>{esc(subtitle)}</p>
    </div>
    """)


def display_footer():
    render_html("""
    <div class="footer">
        <strong>🔬 SkinSense AI</strong> | AI-Based Skin Image Classification
        <br>
        Developed using Python, Streamlit and YOLO.
        <br>
        For educational and research purposes only. Not a substitute for professional medical advice.
    </div>
    """)


def display_card(title, description, emoji):
    render_html(f"""
    <div class="info-card">
        <h3>{emoji} {esc(title)}</h3>
        <p>{esc(description)}</p>
    </div>
    """)


def display_metric(label, value):
    render_html(f"""
    <div class="metric-card">
        <div class="metric-label">{esc(label)}</div>
        <div class="metric-value">{esc(value)}</div>
    </div>
    """)


def display_result_card(label, value, caption):
    render_html(f"""
    <div class="result-card">
        <div class="section-label">{esc(label)}</div>
        <div class="result-title">{esc(value)}</div>
        <div class="result-caption">{esc(caption)}</div>
    </div>
    """)


# ============================================================
# 7. PREDICTION HELPERS
# ============================================================

def normalize_predictions(raw_predictions, disease, confidence):
    results = []

    if isinstance(raw_predictions, dict):
        for name, score in raw_predictions.items():
            try:
                results.append({"Predicted Class": str(name), "Confidence (%)": float(score)})
            except (ValueError, TypeError):
                continue

    elif isinstance(raw_predictions, (list, tuple)):
        for item in raw_predictions:
            try:
                if isinstance(item, dict):
                    name = (
                        item.get("class") or item.get("name")
                        or item.get("disease") or item.get("label")
                    )
                    score = item.get("confidence") if "confidence" in item else item.get("score")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    name, score = item[0], item[1]
                else:
                    continue

                if name is None or score is None:
                    continue
                results.append({"Predicted Class": str(name), "Confidence (%)": float(score)})
            except (ValueError, TypeError):
                continue

    if results and all(0 <= r["Confidence (%)"] <= 1 for r in results):
        for r in results:
            r["Confidence (%)"] *= 100

    if not results:
        results = [{"Predicted Class": disease, "Confidence (%)": confidence}]

    results.sort(key=lambda r: r["Confidence (%)"], reverse=True)
    return results[:5]


def to_list(value):
    """Always return a clean list of strings (fixes one-letter-per-row bullets
    that happen when a plain string is iterated character by character)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = [str(v) for v in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        raw = re.split(r"[\n;•]+", text)
        if len(raw) == 1:
            raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if len(raw) == 1 and "," in text and "." not in text:
            raw = text.split(",")
    items = []
    for item in raw:
        item = re.sub(r"^\s*(?:[-•*·–]|\d+[.)])\s*", "", item).strip()
        if item:
            items.append(item)
    return items


def list_card(title, emoji, items, empty_text="No information available."):
    items = to_list(items)
    if items:
        body = "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"
    else:
        body = f'<p class="empty">{esc(empty_text)}</p>'
    render_html(f'<div class="list-card"><h3>{emoji} {esc(title)}</h3>{body}</div>')


def safe_disease_info(disease):
    info = get_disease_info(disease)
    if not isinstance(info, dict):
        raise ValueError("Disease information must be returned as a dictionary.")
    return {
        "description": str(info.get("description", "Educational information is not available.")),
        "causes": to_list(info.get("causes")),
        "precautions": to_list(info.get("precautions")),
        "symptoms": to_list(info.get("symptoms")),
        "consult": str(info.get("consult", "Consult a qualified dermatologist for medical evaluation.")),
    }


def build_report(result):
    bullets = lambda items: "\n".join(f"- {x}" for x in items)
    top = "\n".join(
        f"- {p['Predicted Class']}: {p['Confidence (%)']:.2f}%"
        for p in result["predictions"]
    )
    return (
        "SKINSENSE AI - CLASSIFICATION REPORT\n"
        "=====================================\n\n"
        f"Analysis Date: {result['timestamp']}\n"
        f"Image: {result['filename']}\n"
        f"Predicted Class: {result['disease']}\n"
        f"Model Confidence: {result['confidence']:.2f}%\n\n"
        f"TOP PREDICTIONS:\n{top}\n\n"
        f"DESCRIPTION:\n{result['info']['description']}\n\n"
        f"POSSIBLE CAUSES:\n{bullets(result['info']['causes'])}\n\n"
        f"GENERAL PRECAUTIONS:\n{bullets(result['info']['precautions'])}\n\n"
        f"COMMON SYMPTOMS:\n{bullets(result['info']['symptoms'])}\n\n"
        f"MEDICAL GUIDANCE:\n{result['info']['consult']}\n\n"
        "IMPORTANT DISCLAIMER:\n"
        "This AI-generated report is for educational purposes only.\n"
        "It is not a medical diagnosis or treatment recommendation.\n"
        "Consult a qualified dermatologist.\n"
    )


def run_analysis(image, image_id, filename):
    predictor = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    prediction = predictor.predict(image)

    if not isinstance(prediction, dict):
        raise ValueError("Unexpected prediction format. Expected a dictionary.")
    for key in ("predicted_disease", "confidence"):
        if key not in prediction:
            raise ValueError(f"Missing prediction field: {key}")

    disease = str(prediction["predicted_disease"])
    confidence = float(prediction["confidence"])

    if confidence <= 1:
        confidence *= 100
    if not 0 <= confidence <= 100:
        raise ValueError("Confidence must be between 0 and 100.")

    info = safe_disease_info(disease)
    all_predictions = normalize_predictions(
        prediction.get("all_predictions", []), disease, confidence
    )
    timestamp = datetime.now().strftime("%d %B %Y, %I:%M:%S %p")

    result = {
        "disease": disease,
        "confidence": confidence,
        "info": info,
        "predictions": all_predictions,
        "timestamp": timestamp,
        "image_id": image_id,
        "filename": filename,
    }
    st.session_state.last_prediction = result
    st.session_state.history.insert(0, {
        "Date": timestamp,
        "Image": filename,
        "Predicted Class": disease,
        "Confidence (%)": round(confidence, 2),
    })


def show_bullets(items, empty_text):
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty_text)


def show_result(result):
    st.markdown("---")
    st.markdown("## 📊 AI Analysis Results")
    st.caption(f"Analysis date: {result['timestamp']}")

    c1, c2 = st.columns(2)
    with c1:
        display_result_card("Predicted Class", result["disease"], "YOLO classification result")
    with c2:
        display_result_card("Model Confidence", f"{result['confidence']:.2f}%", "Classification confidence score")

    st.markdown("### Confidence visualization")
    st.progress(min(max(result["confidence"] / 100, 0), 1))
    st.caption("Model confidence is not the probability that the person has the predicted condition.")

    if result["confidence"] < 50:
        st.warning("The model has relatively low confidence. Consider this result inconclusive.")

    st.markdown("---")
    st.markdown("## Top Predictions")
    df = pd.DataFrame(result["predictions"])
    df["Confidence (%)"] = df["Confidence (%)"].round(2)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("## 📚 Educational Information")
    st.write(result["info"]["description"])

    cause_col, prec_col = st.columns(2, gap="large")
    with cause_col:
        list_card("Possible Causes", "🔎", result["info"]["causes"],
                  "No additional cause information available.")
    with prec_col:
        list_card("General Precautions", "🛡️", result["info"]["precautions"],
                  "No additional precaution information available.")

    list_card("Common Symptoms", "🩺", result["info"]["symptoms"],
              "Symptoms vary and require clinical assessment.")

    st.warning(result["info"]["consult"])
    st.error(
        "Medical disclaimer: This is an AI-generated classification result, "
        "not a confirmed diagnosis. Consult a qualified dermatologist for medical advice."
    )

    st.download_button(
        "📥 Download Analysis Report",
        data=build_report(result),
        file_name="SkinSense_AI_Report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:
    render_html("""
    <div style="padding:10px 0 20px 0;">
        <h1 style="color:white;font-size:27px;">🔬 SkinSense AI</h1>
        <p style="color:#A9CAD8;font-size:12px;">INTELLIGENT SKIN ANALYSIS</p>
    </div>
    """)
    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        ["Home", "Skin Analysis", "Prediction History", "Skin Care Guide"],
        key="navigation",
    )

    st.markdown("---")
    # # st.markdown("### 🧠 AI MODEL")
    # # st.write("YOLO Classification")
    # if MODEL_PATH.exists():
    #     st.success("Model file detected")
    # else:
    #     st.error("Model file not found")

    st.markdown("---")
    st.caption(f"Version {APP_VERSION}")
    st.caption("AI-powered educational prototype")


# ============================================================
# 9. HOME PAGE
# ============================================================

if page == "Home":
    display_header(
        "Welcome to SkinSense AI",
        "Explore AI-powered skin image classification and educational information about skin conditions.",
    )

    st.markdown("## Discover AI-powered skin analysis")
    st.write(
        "SkinSense AI is a machine-learning-based research application designed to "
        "classify uploaded skin images using a trained YOLO classification model. "
        "It provides prediction results, model confidence scores and educational "
        "information about the predicted class."
    )

    st.markdown("---")
    st.markdown("## Application Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        display_card("AI Classification", "Analyze uploaded skin images using a trained YOLO classification model.", "🧠")
    with c2:
        display_card("Prediction Analysis", "View the predicted class, confidence score and alternative predictions.", "📊")
    with c3:
        display_card("Skin Education", "Explore educational information, possible causes, symptoms and general precautions.", "📚")

    st.markdown("---")
    st.markdown("## How SkinSense AI works")
    step_data = [
        ("01", "Upload Image", "Select a clear skin image."),
        ("02", "AI Processing", "YOLO processes the image."),
        ("03", "View Results", "Review the classification."),
        ("04", "Learn More", "Explore educational information."),
    ]
    for col, (num, title, desc) in zip(st.columns(4), step_data):
        with col:
            render_html(f"<h2 style='color:#176B87'>{num}</h2>")
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("---")
    st.markdown("## Ready to analyze an image?")
    st.write("Upload an image and explore the classification results generated by the trained AI model.")
    st.button("🔍 Start Skin Analysis", type="primary", use_container_width=True, on_click=go_to_analysis)
    st.info("This application is for educational and research purposes. AI predictions are not medical diagnoses.")


# ============================================================
# 10. SKIN ANALYSIS PAGE
# ============================================================

elif page == "Skin Analysis":
    display_header("Skin Image Analysis", "Upload a skin image to generate an AI-based classification.")

    st.markdown("## Upload your image")
    st.write("Supported formats: JPG, JPEG and PNG. Use a clear, well-lit image for better results.")

    uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"], key="skin_image")

    if uploaded_file is None:
        st.info("Upload a skin image to begin the AI classification.")
        display_card(
            "Image Upload Guidelines",
            "Upload JPG, JPEG or PNG images. Use clear and well-lit images. "
            f"Avoid blurry or dark images. Maximum size: {MAX_IMAGE_SIZE_MB} MB.",
            "📌",
        )

    elif uploaded_file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        st.error(f"Image exceeds the {MAX_IMAGE_SIZE_MB} MB limit.")

    else:
        try:
            image_bytes = uploaded_file.getvalue()
            image_id = hashlib.md5(image_bytes).hexdigest()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            st.markdown("---")
            image_col, details_col = st.columns([1.1, 1], gap="large")

            with image_col:
                st.markdown("### Image Preview")
                st.image(image, use_container_width=True)

            with details_col:
                st.markdown("### Image Information")
                display_metric("Image Width", f"{image.width} px")
                st.write("")
                display_metric("Image Height", f"{image.height} px")
                st.write("")
                st.write(f"**File name:** {uploaded_file.name}")
                st.write("**Image format:** RGB")
                st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")

            st.markdown("---")
            analyze = st.button(
                "🔍 Analyze Skin Image", type="primary",
                use_container_width=True, key="analyze_button",
            )

            if analyze:
                if not MODEL_PATH.exists():
                    st.error("YOLO model not found. Place best.pt inside the models folder.")
                else:
                    try:
                        with st.spinner("AI model is analyzing your image. Please wait..."):
                            run_analysis(image, image_id, uploaded_file.name)
                        st.success("Image classification completed successfully!")
                    except Exception as error:
                        st.session_state.last_prediction = None
                        st.error("An error occurred during image analysis.")
                        st.exception(error)

            result = st.session_state.last_prediction
            if result is not None and result.get("image_id") == image_id:
                show_result(result)

        except Exception as error:
            st.error(f"Unable to process the uploaded image: {error}")


# ============================================================
# 11. PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":
    display_header("Prediction History", "Review and manage your previous AI classification results.")

    history = st.session_state.history

    if history:
        df = pd.DataFrame(history)

        c1, c2 = st.columns(2)
        with c1:
            display_metric("Total Predictions", len(history))
        with c2:
            display_metric("Unique Classes", df["Predicted Class"].nunique())

        st.markdown("---")
        st.markdown("## Previous Analyses")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download History CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="SkinSense_Prediction_History.csv",
            mime="text/csv",
        )

        st.markdown("---")
        if st.button("🗑️ Clear Prediction History", type="secondary"):
            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()
    else:
        st.info("No prediction history available. Complete your first skin image analysis to see results here.")


# ============================================================
# 12. SKIN CARE GUIDE PAGE
# ============================================================

elif page == "Skin Care Guide":
    display_header(
        "Skin Care Guide",
        "Simple, practical habits to keep the skin on your whole body healthy.",
    )

    st.markdown("## Why skin care matters")
    st.write(
        "Your skin is the body's largest organ. It protects you from germs, sunlight, "
        "heat and injury, helps regulate temperature and lets you feel touch. Small daily "
        "habits keep this barrier strong and help prevent many common skin problems."
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        display_metric("Daily sunscreen", "SPF 30+")
    with m2:
        display_metric("Ideal bath time", "5–10 min")
    with m3:
        display_metric("Sleep for adults", "7–9 hrs")

    st.write("")

    tab_routine, tab_types, tab_habits, tab_body, tab_doctor = st.tabs([
        "🧴 Daily Routine", "🔬 Skin Types", "🥗 Healthy Habits",
        "🧍 Body Skin Care", "🩺 When to See a Doctor",
    ])

    with tab_routine:
        morning, evening = st.columns(2, gap="large")
        with morning:
            list_card("Morning Routine", "🌅", [
                "Cleanse gently with lukewarm water and a mild, fragrance-free cleanser.",
                "Apply a light moisturizer suited to your skin type, even if your skin is oily.",
                "Use broad-spectrum sunscreen (SPF 30 or higher) on all exposed skin: face, neck, ears and hands.",
                "Wear a hat, sunglasses and light full-sleeve clothing if you will be outdoors for long.",
            ])
        with evening:
            list_card("Evening Routine", "🌙", [
                "Wash off sweat, dust, sunscreen and makeup with a gentle cleanser.",
                "Pat the skin dry with a clean towel instead of rubbing.",
                "Moisturize while the skin is slightly damp to lock in moisture.",
                "Add only one new product at a time and patch-test it on a small area first.",
            ])
        list_card("Weekly Care", "📅", [
            "Exfoliate gently no more than once or twice a week, and avoid harsh scrubs.",
            "Change pillowcases and bed sheets weekly, and wash towels regularly.",
            "Keep nails short and clean to avoid scratching and infection.",
            "Check your skin once a month, head to toe, for new or changing spots and moles.",
        ])
        list_card("Common Mistakes to Avoid", "⚠️", [
            "Over-washing or over-scrubbing, which damages the skin barrier.",
            "Picking or popping pimples, which can cause scars and infection.",
            "Skipping sunscreen on cloudy days or indoors near windows.",
            "Sleeping with makeup on or using many new products at once.",
            "Sharing towels, razors or skin products with others.",
        ])

    with tab_types:
        st.write("Knowing your skin type helps you choose the right products.")
        a, b, c = st.columns(3, gap="large")
        with a:
            list_card("Normal Skin", "🙂", [
                "Feels balanced, neither too oily nor too dry.",
                "Use a gentle cleanser and a light moisturizer.",
                "Daily sunscreen is still essential.",
            ])
        with b:
            list_card("Dry Skin", "🏜️", [
                "Feels tight, rough or flaky, especially after bathing.",
                "Use a creamy cleanser and a rich moisturizer with ingredients like glycerin or ceramides.",
                "Avoid hot water and strong soaps.",
            ])
        with c:
            list_card("Oily Skin", "💧", [
                "Looks shiny, with larger pores and a tendency to break out.",
                "Use a gel-based cleanser and an oil-free, non-comedogenic moisturizer.",
                "Do not over-wash, as it can make skin produce more oil.",
            ])
        d, e = st.columns(2, gap="large")
        with d:
            list_card("Combination Skin", "⚖️", [
                "Oily forehead, nose and chin, with normal or dry cheeks.",
                "Use a gentle cleanser and a light, non-greasy moisturizer.",
                "Apply richer products only to the dry areas.",
            ])
        with e:
            list_card("Sensitive Skin", "🌸", [
                "Stings, burns or turns red easily with new products.",
                "Choose fragrance-free, minimal-ingredient products.",
                "Always patch-test before using something new.",
            ])

    with tab_habits:
        h1, h2 = st.columns(2, gap="large")
        with h1:
            list_card("Eat for Healthy Skin", "🥗", [
                "Eat plenty of fruits, vegetables, whole grains and nuts.",
                "Include healthy fats such as fish, flaxseed and walnuts.",
                "Limit sugary drinks and heavily processed foods.",
            ])
            list_card("Sleep and Stress", "😴", [
                "Aim for 7–9 hours of sleep so the skin can repair itself.",
                "Stress can trigger flare-ups of acne, eczema and psoriasis, so make time to relax.",
                "Regular exercise improves circulation and mood. Shower soon after sweating.",
            ])
        with h2:
            list_card("Sun Safety", "☀️", [
                "Avoid direct sun in the middle of the day, roughly 10 am to 4 pm.",
                "Reapply sunscreen about every 2 hours, and after swimming or heavy sweating.",
                "Seek shade and wear protective clothing.",
                "Avoid tanning beds completely.",
            ])
            list_card("Lifestyle Choices", "🚭", [
                "Do not smoke. Smoking speeds up skin ageing and slows healing.",
                "Limit alcohol, which can dry out the skin.",
                "Drink water regularly. It supports overall health, but a moisturizer is what protects the skin barrier.",
            ])

    with tab_body:
        b1, b2 = st.columns(2, gap="large")
        with b1:
            list_card("Bathing", "🚿", [
                "Take lukewarm showers of about 5–10 minutes. Very hot water dries the skin.",
                "Use mild soap mainly on underarms, groin and feet, and plain water elsewhere.",
                "Pat dry well, especially between the toes and in skin folds.",
                "Moisturize within a few minutes of bathing. Use thicker creams on elbows, knees and heels.",
            ])
            list_card("Hands and Feet", "🖐️", [
                "Wash hands with soap and water, then moisturize. Wear gloves for dishes and cleaning.",
                "Dry between the toes and wear clean socks and breathable footwear to help prevent fungal infections.",
                "Avoid walking barefoot in wet public areas such as locker rooms and pool sides.",
            ])
        with b2:
            list_card("Clothing and Hygiene", "👕", [
                "Wear loose, breathable cotton clothes, especially in hot and humid weather.",
                "Change out of sweaty or wet clothes promptly.",
                "Do not share towels, combs, razors or clothing.",
                "Wash new clothes before wearing them and keep bedding clean.",
            ])
            list_card("Cuts, Bites and Shaving", "🩹", [
                "Clean small cuts with clean water and cover with a clean dressing. Do not pick scabs.",
                "Use insect repellent and cover exposed skin where insects are common. Avoid scratching bites.",
                "Shave with a clean, sharp razor and shaving gel, in the direction of hair growth.",
            ])

    with tab_doctor:
        st.write("See a dermatologist or doctor if you notice any of the following.")
        list_card("Get Medical Advice If", "🩺", [
            "A mole or spot changes in size, shape or colour, has uneven edges, itches, bleeds or looks different from your other moles.",
            "A sore or wound does not heal within a few weeks.",
            "A rash spreads quickly, is painful, forms blisters or comes with fever.",
            "There are signs of infection: increasing redness, warmth, swelling, pus or red streaks.",
            "Itching keeps you awake or a rash does not improve with basic care after one to two weeks.",
        ])
        list_card("Avoid Self-Treatment", "🚫", [
            "Do not use steroid or combination creams without a doctor's advice. They can worsen fungal infections and thin the skin.",
            "Do not use leftover prescription creams or home remedies on a rash you cannot identify.",
        ])
        st.error(
            "Seek emergency care immediately if a rash comes with swelling of the face, lips or "
            "tongue, or difficulty breathing."
        )

    st.info(
        "This guide gives general educational information and is not a substitute for "
        "professional medical advice. Consult a qualified dermatologist for personal concerns."
    )

display_footer()