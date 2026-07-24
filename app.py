import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="UNO Card Detector",
    page_icon="🃏",
    layout="wide"
)

# ---------------- CSS ----------------

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# ---------------- Sidebar ----------------

with st.sidebar:

    st.image("logo.jpeg", use_container_width=True)

    st.title("🃏 UNO AI")

    st.write("---")

    st.info("""
Upload an image of an UNO card.

The AI can detect:

✅ Number Cards

✅ Skip

✅ Reverse

✅ Draw Two

✅ Wild

✅ Wild Draw Four
""")

    st.write("---")

    st.success("Developed by")

    st.markdown("### **Madishetty Sriraj**")

    st.caption("YOLOv8 • Streamlit • Python")

# ---------------- Header ----------------

st.markdown("""
<div class="title">
<h1>🃏 UNO CARD DETECTOR</h1>
<h3>Developed by <span>Madishetty Sriraj</span></h3>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- Load Model ----------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- Upload ----------------

uploaded_file = st.file_uploader(
    "📤 Upload an UNO Card",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

if image.mode != "RGB":
    image = image.convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 📤 Uploaded Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    with st.spinner("🤖 AI is detecting the card..."):

        results = model.predict(
            source=temp_path,
            conf=0.25
        )

    result = results[0]

    annotated = result.plot()
    annotated = Image.fromarray(annotated[:, :, ::-1])

    with col2:
        st.markdown("## 🎯 Detection Result")
        st.image(annotated, use_container_width=True)

    st.divider()

    st.markdown("## 🎯 AI Prediction")

    if len(result.boxes) == 0:

        st.error("❌ No UNO Card Detected")

    else:

        st.balloons()

        top_card = ""
        top_conf = 0

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            card = model.names[cls]

            if conf > top_conf:
                top_conf = conf
                top_card = card

            c1, c2 = st.columns([4,1])

            with c1:
                st.success(f"🃏 **{card}**")

            with c2:
                st.metric("Confidence", f"{conf*100:.1f}%")

            st.progress(conf)

        st.divider()

        st.subheader("📊 Detection Summary")

        a, b, c = st.columns(3)

        a.metric("Cards Found", len(result.boxes))
        b.metric("Top Prediction", top_card)
        c.metric("Best Confidence", f"{top_conf*100:.2f}%")

    annotated.save("prediction.jpg")

    with open("prediction.jpg", "rb") as file:

        st.download_button(
            "📥 Download Prediction",
            file,
            file_name="prediction.jpg",
            mime="image/jpeg"
        )

st.write("")
st.write("")

st.markdown("""
<hr>
<center>

<h4>Made with ❤️ using YOLOv8</h4>

<h3>© 2026 Madishetty Sriraj</h3>

</center>
""", unsafe_allow_html=True)
