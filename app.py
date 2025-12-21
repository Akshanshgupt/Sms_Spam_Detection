import streamlit as st
import pickle
import numpy as np

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# ===================== LOAD MODEL & VECTORIZER =====================
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ===================== UI =====================
st.title("📩 Spam Message Detector")
st.write("Check whether a message is **Spam** or **Ham**")

message = st.text_area(
    "✉️ Enter your message",
    height=150,
    placeholder="Congratulations! You won a free prize..."
)

if st.button("🔍 Check Message"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Transform message
        message_vector = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_vector)[0]

        # Probability (if available)
        try:
            probability = model.predict_proba(message_vector)
            confidence = np.max(probability) * 100
        except:
            confidence = None

        # Display result
        if prediction.lower() == "spam" or prediction == 1:
            st.error("🚨 **SPAM MESSAGE**")
        else:
            st.success("✅ **HAM (NOT SPAM)**")

        if confidence:
            st.info(f"📊 Confidence: **{confidence:.2f}%**")

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Spam Detection using Machine Learning & Streamlit")
