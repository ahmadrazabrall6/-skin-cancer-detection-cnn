
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Skin Cancer AI Detection",
    page_icon="🔬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}
.title {
    text-align: center;
    color: #12345a;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #52677d;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# Heading
st.markdown(
    '<div class="title">🔬 AI Skin Cancer Classification</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">CNN Based Skin Lesion Image Classification</div>',
    unsafe_allow_html=True
)

st.divider()

# Class Labels
class_names = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)"
]

# Load Model
@st.cache_resource
def load_skin_model():
    return load_model("skin_cancer_final_model.keras")

try:
    model = load_skin_model()
    st.success("AI Model Loaded Successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Image Upload
st.subheader("📤 Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Skin Image",
            use_container_width=True
        )

    with col2:
        st.subheader("🔍 Classification Results")

        if st.button("Analyze Image", type="primary"):

            with st.spinner("AI is analyzing the image..."):

                # Preprocessing
                resized_image = image.resize((96, 96))

                img_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(predictions[0])
                )

                confidence = float(
                    predictions[0][predicted_index] * 100
                )

                st.success("Analysis Completed!")

                st.metric(
                    "Predicted Class",
                    class_names[predicted_index]
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Class Probabilities")

                for i, name in enumerate(class_names):
                    probability = float(
                        predictions[0][i] * 100
                    )
                    st.write(f"**{name}**")
                    st.progress(
                        min(max(probability / 100, 0.0), 1.0)
                    )
                    st.caption(f"{probability:.2f}%")

st.divider()

st.info(
    "Educational project only. This AI model is not a medical "
    "diagnostic tool. Results must not be used to diagnose or "
    "exclude skin cancer. Consult a qualified dermatologist."
)

st.caption("CNN Based Skin Cancer Classification | AI/ML Project")
