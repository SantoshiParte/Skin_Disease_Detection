
import streamlit as st

st.set_page_config(
    page_title="AI Skin Disease Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Skin Disease Analyzer")

st.write("Welcome to the Skin Disease Detection System.")

st.markdown("---")

st.header("Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Skin Image",
        use_container_width=True
    )

    if st.button("Analyze Image"):

        st.success("Image uploaded successfully!")

st.markdown("---")

st.info("YOLO model integration will be added here.")
