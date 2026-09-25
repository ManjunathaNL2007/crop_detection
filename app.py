from datetime import datetime
from knowledge import knowledge
import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = load_model("crop_disease_model.h5")

# ---------------- CLASS NAMES ----------------

class_names = [
    "Potato Common Scab",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight"
]

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌱 Crop AI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Disease Detection",
        "Prediction History",
        "AI Assistant",
        "About Project"
    ]
)


# ---------------- LANGUAGE SELECTOR ----------------

st.markdown(
    "<h3 style='color:#7dff8f;'>🌍 Choose Language</h3>",
    unsafe_allow_html=True
)

language = st.selectbox(
    "",
    ["English", "Kannada", "Hindi","Telugu","Tamil"]
)

# ---------------- TRANSLATIONS ----------------
translations = {

    "English": {

        "Potato Common Scab": "Potato Common Scab",
        "Potato Early Blight": "Potato Early Blight",
        "Potato Healthy": "Potato Healthy",
        "Potato Late Blight": "Potato Late Blight",

        "history": "Prediction History",
        "date": "Date & Time",
        "crop": "Crop",
        "disease": "Disease",
        "confidence": "Confidence",
        "image": "Image",
        "download_history": "Download History",
        "clear_history": "Clear History",
        "no_history": "No prediction history available."

    },

    "Kannada": {

        "Potato Common Scab": "ಆಲೂಗಡ್ಡೆ ಕಾಮನ್ ಸ್ಕ್ಯಾಬ್",
        "Potato Early Blight": "ಆಲೂಗಡ್ಡೆ ಅರ್ಲಿ ಬ್ಲೈಟ್",
        "Potato Healthy": "ಆರೋಗ್ಯಕರ ಆಲೂಗಡ್ಡೆ",
        "Potato Late Blight": "ಆಲೂಗಡ್ಡೆ ಲೇಟ್ ಬ್ಲೈಟ್",

        "history": "ಭವಿಷ್ಯವಾಣಿ ಇತಿಹಾಸ",
        "date": "ದಿನಾಂಕ ಮತ್ತು ಸಮಯ",
        "crop": "ಬೆಳೆ",
        "disease": "ರೋಗ",
        "confidence": "ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "image": "ಚಿತ್ರ",
        "download_history": "ಇತಿಹಾಸ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "clear_history": "ಇತಿಹಾಸ ಅಳಿಸಿ",
        "no_history": "ಯಾವುದೇ ಇತಿಹಾಸ ಲಭ್ಯವಿಲ್ಲ."

    },

    "Hindi": {

        "Potato Common Scab": "आलू कॉमन स्कैब",
        "Potato Early Blight": "आलू अर्ली ब्लाइट",
        "Potato Healthy": "स्वस्थ आलू",
        "Potato Late Blight": "आलू लेट ब्लाइट",

        "history": "भविष्यवाणी इतिहास",
        "date": "दिनांक और समय",
        "crop": "फसल",
        "disease": "रोग",
        "confidence": "विश्वास स्तर",
        "image": "चित्र",
        "download_history": "इतिहास डाउनलोड करें",
        "clear_history": "इतिहास हटाएँ",
        "no_history": "कोई इतिहास उपलब्ध नहीं है।"

    },

    "Telugu": {

        "Potato Common Scab": "ఆలు కామన్ స్కాబ్",
        "Potato Early Blight": "ఆలు ఎర్లీ బ్లైట్",
        "Potato Healthy": "ఆలు హెల్తి",
        "Potato Late Blight": "ఆలు లేట్ బ్లైట్",

        "history": "అంచనా చరిత్ర",
        "date": "తేదీ మరియు సమయం",
        "crop": "పంట",
        "disease": "వ్యాధి",
        "confidence": "నమ్మకం స్థాయి",
        "image": "చిత్రం",
        "download_history": "చరిత్రను డౌన్‌లోడ్ చేయండి",
        "clear_history": "చరిత్రను తొలగించండి",
        "no_history": "అంచనా చరిత్ర అందుబాటులో లేదు."

    },

    "Tamil": {

        "Potato Common Scab": "ஆலு காமன் ஸ்கேப்",
        "Potato Early Blight": "ஆலு எர்லி ப்ளைட்",
        "Potato Healthy": "ஆலு சுகாதாரமான",
        "Potato Late Blight": "ஆலு லேட் ப்ளைட்",

        "history": "கணிப்பு வரலாறு",
        "date": "தேதி மற்றும் நேரம்",
        "crop": "பயிர்",
        "disease": "நோய்",
        "confidence": "நம்பிக்கை அளவு",
        "image": "படம்",
        "download_history": "வரலாற்றைப் பதிவிறக்கவும்",
        "clear_history": "வரலாற்றை அழிக்கவும்",
        "no_history": "கணிப்பு வரலாறு இல்லை."

    }

}
t = translations[language]

    

# ---------------- SAVE HISTORY ----------------

def save_prediction(crop, disease, confidence, image_name):

    history_file = "prediction_history.csv"

    data = pd.DataFrame({
        "Date & Time":[datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        "Crop":[crop],
        "Disease":[disease],
        "Confidence":[f"{confidence:.2f}%"],
        "Image":[image_name]
    })

    if os.path.exists(history_file):
        data.to_csv(history_file,mode="a",header=False,index=False)

    else:
        data.to_csv(history_file,index=False)
def translate(text):

    translations = {

        "English": {
            "Cause": "Cause",
            "Symptoms": "Symptoms",
            "Treatment": "Treatment",
            "Prevention": "Prevention",
            "Fertilizer": "Fertilizer",
            "Recovery Time": "Recovery Time",
            "Current Analysis": "Current Analysis",
            "Disease": "Disease",
            "Confidence": "Confidence",
            "Health Score": "Health Score",
            "No information available.": "No information available.",
            "Please detect a crop disease first.": "Please detect a crop disease first."
        },

        "Kannada": {
            "Cause": "ಕಾರಣ",
            "Symptoms": "ಲಕ್ಷಣಗಳು",
            "Treatment": "ಚಿಕಿತ್ಸೆ",
            "Prevention": "ತಡೆಗಟ್ಟುವಿಕೆ",
            "Fertilizer": "ರಸಗೊಬ್ಬರ",
            "Recovery Time": "ಚೇತರಿಕೆಯ ಸಮಯ",
            "Current Analysis": "ಪ್ರಸ್ತುತ ವಿಶ್ಲೇಷಣೆ",
            "Disease": "ರೋಗ",
            "Confidence": "ವಿಶ್ವಾಸ ಮಟ್ಟ",
            "Health Score": "ಆರೋಗ್ಯ ಅಂಕ",
            "No information available.": "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.",
            "Please detect a crop disease first.": "ಮೊದಲು ಬೆಳೆ ರೋಗವನ್ನು ಪತ್ತೆಹಚ್ಚಿ."
        },

        "Hindi": {
            "Cause": "कारण",
            "Symptoms": "लक्षण",
            "Treatment": "उपचार",
            "Prevention": "रोकथाम",
            "Fertilizer": "उर्वरक",
            "Recovery Time": "ठीक होने का समय",
            "Current Analysis": "वर्तमान विश्लेषण",
            "Disease": "रोग",
            "Confidence": "विश्वास स्तर",
            "Health Score": "स्वास्थ्य स्कोर",
            "No information available.": "जानकारी उपलब्ध नहीं है।",
            "Please detect a crop disease first.": "कृपया पहले फसल रोग का पता लगाएं।"
        },

        "Telugu": {
            "Cause": "కారణం",
            "Symptoms": "లక్షణాలు",
            "Treatment": "చికిత్స",
            "Prevention": "నివారణ",
            "Fertilizer": "ఎరువు",
            "Recovery Time": "కోలుకునే సమయం",
            "Current Analysis": "ప్రస్తుత విశ్లేషణ",
            "Disease": "వ్యాధి",
            "Confidence": "నమ్మకం",
            "Health Score": "ఆరోగ్య స్కోర్",
            "No information available.": "సమాచారం అందుబాటులో లేదు.",
            "Please detect a crop disease first.": "దయచేసి ముందుగా పంట వ్యాధిని గుర్తించండి."
        },

        "Tamil": {
            "Cause": "காரணம்",
            "Symptoms": "அறிகுறிகள்",
            "Treatment": "சிகிச்சை",
            "Prevention": "தடுப்பு",
            "Fertilizer": "உரம்",
            "Recovery Time": "மீட்பு நேரம்",
            "Current Analysis": "தற்போதைய பகுப்பாய்வு",
            "Disease": "நோய்",
            "Confidence": "நம்பிக்கை",
            "Health Score": "ஆரோக்கிய மதிப்பெண்",
            "No information available.": "தகவல் கிடைக்கவில்லை.",
            "Please detect a crop disease first.": "முதலில் பயிர் நோயைக் கண்டறியவும்."
        }

    }

    return translations.get(language, translations["English"]).get(text, text)

def get_ai_response(question):
    lang = language

    disease = st.session_state.get("current_disease", "Unknown")
    confidence = st.session_state.get("current_confidence", 0)
    health = st.session_state.get("current_health_score", 0)

    if disease == "Unknown":
        return "⚠ Please detect a crop disease first."

    info = knowledge.get(disease, {}).get(lang, {})

    if not info:
       return f"No information available in {language}."
    
    question = question.lower()

    answer = f"""
## 🌿 {translate("Current Analysis")}

🦠 **{translate("Disease")}:** {disease}

🎯 **{translate("Confidence")}:** {confidence:.2f}%

💚 **{translate("Health Score")}:** {health}/100

"""
    

    if "cause" in question:

        answer += f"""

### 🦠 {translate("Cause")}
{info["cause"]}
"""

    elif "symptom" in question:

        answer += f"### 🌿 {translate('Symptoms')}\n\n"

        for s in info["symptoms"]:
            answer += f"• {s}\n"

    elif "treat" in question or "solution" in question:

        answer += f"### 💊 {translate('Treatment')}\n\n"

        for t in info["treatment"]:
            answer += f"✅ {t}\n"

    elif "prevent" in question:

        answer += f"### 🛡 {translate('Prevention')}\n\n"

        for p in info["prevention"]:
            answer += f"✅ {p}\n"

    elif "organic" in question:

        answer += f"### 🌱 {translate('Organic Solution')}\n\n"

        for o in info["organic"]:
            answer += f"🌿 {o}\n"

    elif "fertilizer" in question:

        answer += f"""

### 🧪 {translate('Fertilizer')}

{info["fertilizer"]}
"""

    elif "recovery" in question:

        answer += f"""

### ⏳ {translate('Recovery Time')} 

{info["recovery"]}
"""

    else:

        answer += """
You can ask me:

• Cause

• Symptoms

• Treatment

• Prevention

• Organic Solution

• Fertilizer

• Recovery Time
"""

    return answer

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #04130b, #071d12, #04130b);
    color: white;
}

.hero {
    padding: 45px;
    border-radius: 25px;
    background: rgba(0,255,100,0.05);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(0,255,100,0.15);
    margin-bottom: 35px;
    box-shadow: 0px 0px 25px rgba(0,255,100,0.08);
}

.hero-title {
    font-size: 52px;
    font-weight: bold;
    color: #7dff8f;
    margin-bottom: 15px;
}

.hero-subtitle {
    font-size: 22px;
    color: #d8ffe0;
    line-height: 1.5;
}

.feature-card {
    background: rgba(0,255,100,0.04);
    padding: 28px;
    border-radius: 22px;
    border: 1px solid rgba(0,255,100,0.12);
    box-shadow: 0px 0px 20px rgba(0,255,100,0.05);
}

.result-card {
    background: rgba(0,255,100,0.04);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(0,255,100,0.12);
}

.chat-box {
    background: rgba(0,255,100,0.04);
    padding: 25px;
    border-radius: 22px;
    border: 1px solid rgba(0,255,100,0.12);
}

.stat-card {
    background: rgba(0,255,100,0.04);
    padding: 25px;
    border-radius: 22px;
    text-align: center;
    border: 1px solid rgba(0,255,100,0.12);
    box-shadow: 0px 0px 15px rgba(0,255,100,0.05);
}

.stat-title {
    font-size: 40px;
    color: #7dff8f;
    font-weight: bold;
}

.stat-text {
    font-size: 18px;
    color: #d8ffe0;
}

h1,h2,h3,h4 {
    color: white;
}
            div[role="radiogroup"] label p {
    color: #7dff8f !important;
    font-size: 20px !important;
    font-weight: bold !important;
    text-shadow: 0 0 10px #7dff8f;
}

[data-testid="stFileUploader"] label {
    color: #7dff8f !important;
    font-size: 20px !important;
    font-weight: bold !important;
    text-shadow: 0 0 10px #7dff8f;
            
}
/* AI Assistant chat text */
[data-testid="stChatMessageContent"] {
    color: #F5FFF5 !important;
}

[data-testid="stChatMessageContent"] * {
    color: #F5FFF5 !important;
}
/* Download button */
.stDownloadButton > button {
    background-color: #2E8B57 !important;
    color: white !important;
    border: 2px solid #66FF99 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-size: 16px !important;
    font-weight: bold !important;
}

/* Style for all Streamlit buttons */
.stButton > button,
.stDownloadButton > button {

    background-color: #2E8B57 !important;
    color: white !important;

    border: 2px solid #66FF99 !important;
    border-radius: 10px !important;

    padding: 10px 18px !important;

    font-size: 16px !important;
    font-weight: bold !important;

    transition: all 0.3s ease;
}

/* Hover effect */
.stButton > button:hover,
.stDownloadButton > button:hover {

    background-color: #3CB371 !important;
    border-color: #98FB98 !important;
    color: white !important;
}

/* Button width */
.stButton > button,
.stDownloadButton > button {
    width: 100%;
</style>
""", unsafe_allow_html=True)

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.markdown("""
    <div class="hero">

    <div class="hero-title">
    🌱 AI-Powered Crop Disease Detection
    </div>

    <div class="hero-subtitle">
    Integrating cutting-edge deep learning algorithms and computer vision technologies.
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("🧠 How It Works")

    st.write("""
1️⃣ Upload potato leaf image  
2️⃣ AI analyzes crop disease  
3️⃣ Disease prediction generated  
4️⃣ Get treatment recommendation  
5️⃣ Download AI report  
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">🤖 4</div>
            <div class="stat-text">Diseases Supported</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">🎯 92%</div>
            <div class="stat-text">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">⚡ AI CNN</div>
            <div class="stat-text">Deep Learning Model</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# DISEASE DETECTION PAGE
# =========================================================

elif page == "Disease Detection":
     st.markdown(
    "<h3 style='color:#7dff8f;'>📷 Choose Image Source</h3>",
    unsafe_allow_html=True
)

     option = st.radio(
    "",
    ["Upload Image", "Use Camera"]
)

     if option == "Upload Image":

        uploaded_file = st.file_uploader(
            "Choose image",
            type=["jpg", "jpeg", "png"]
    )

     else:

       uploaded_file = st.camera_input(
        "Take a photo of the crop leaf"
    )
    
     if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                img,
                caption="Uploaded Leaf",
                width=250
            )

        img_resized = img.resize((128,128))

        img_array = image.img_to_array(img_resized)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0

        with st.spinner("🤖 AI is analyzing crop image..."):

            prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        predicted_disease = class_names[predicted_class]

        translated_disease = translations[language][predicted_disease]

        # ---------------- INVALID IMAGE CHECK ----------------

        if confidence < 55:

            st.error(
                "❌ Invalid image. Please upload a clear potato leaf image."
            )

            st.stop()

        save_prediction(
               crop="Potato",
               disease=predicted_disease,
             confidence=confidence,
             image_name=uploaded_file.name
  )    

        # ---------------- SMART HEALTH SCORE ----------------

        if predicted_disease == "Potato Healthy":

            health_score = int(confidence)

            severity = "Healthy 🟢"

        elif predicted_disease == "Potato Early Blight":

            health_score = int(100 - (confidence * 0.45))

            severity = "Moderate 🟡"

        elif predicted_disease == "Potato Common Scab":

            health_score = int(100 - (confidence * 0.35))

            severity = "Mild 🟠"

        elif predicted_disease == "Potato Late Blight":

            health_score = int(100 - (confidence * 0.60))

            severity = "Severe 🔴"

        # LIMIT VALUES

        health_score = max(10, min(100, health_score))
        st.session_state["current_disease"] = predicted_disease
        st.session_state["current_confidence"] = confidence
        st.session_state["current_health_score"] = health_score

        # ---------------- HEALTH STATUS ----------------

        if health_score >= 80:

            health_status = "Excellent 🟢"

        elif health_score >= 60:

            health_status = "Good 🟡"

        elif health_score >= 40:

            health_status = "Poor 🟠"

        else:

            health_status = "Critical 🔴"

        # ---------------- RECOMMENDATIONS ----------------

        recommendation = ""

        if predicted_disease == "Potato Early Blight":

            recommendation = """
✅ Remove infected leaves
✅ Apply copper fungicide
✅ Maintain proper irrigation
✅ Use crop rotation
"""

        elif predicted_disease == "Potato Late Blight":

            recommendation = """
✅ Apply Mancozeb fungicide
✅ Improve airflow around plants
✅ Remove infected plants immediately
✅ Avoid excess watering
"""

        elif predicted_disease == "Potato Common Scab":

            recommendation = """
✅ Maintain soil moisture
✅ Use certified seeds
✅ Reduce soil pH
✅ Rotate crops regularly
"""

        elif predicted_disease == "Potato Healthy":

            recommendation = """
✅ Plant appears healthy
✅ Continue proper irrigation
✅ Monitor regularly for disease
"""

        with col2:

            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            st.subheader("🌿 Prediction Result")

            st.success(
                f"Disease: {translated_disease}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            st.info(f"🕒 Analysis Time: {current_time}")

            st.progress(int(confidence))

            st.warning(
                f"⚠️ Severity Level: {severity}"
            )

            st.subheader("🌱 Plant Health Score")

            st.progress(health_score)

            st.success(f"{health_score}/100")

            st.info(
                f"Plant Health Status: {health_status}"
            )

            st.subheader("💊 Recommended Solution")

            st.write(recommendation)

            # ---------------- PROBABILITY CHART ----------------

            st.subheader("📊 Disease Probability Analysis")

            probabilities = prediction[0] * 100

            chart_data = pd.DataFrame({
                "Disease": class_names,
                "Probability": probabilities
            })

            st.bar_chart(
                chart_data.set_index("Disease")
            )

            # ---------------- DOWNLOAD REPORT ----------------

            report = f"""
AI Crop Disease Detection Report

Disease:
{translated_disease}

Confidence:
{confidence:.2f}%

Severity:
{severity}

Health Status:
{health_status}

Analysis Time:
{current_time}

Recommended Solution:
{recommendation}

Plant Health Score:
{health_score}/100
"""

            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name="crop_disease_report.txt",
                mime="text/plain"
            )

            st.markdown("</div>", unsafe_allow_html=True)
### =========================================================
            # PREDICTION HISTORY PAGE
### =========================================================
elif page == "Prediction History":

    st.title("📜 " + t["history"])

    if os.path.exists("prediction_history.csv"):

        history = pd.read_csv("prediction_history.csv")
        history["Disease"] = history["Disease"].replace({
             "Potato Common Scab": translations[language]["Potato Common Scab"],
             "Potato Early Blight": translations[language]["Potato Early Blight"],
             "Potato Healthy": translations[language]["Potato Healthy"],
             "Potato Late Blight": translations[language]["Potato Late Blight"]
 }) 
        history["Crop"] = history["Crop"].replace({
    "Potato": {
        "English": "Potato",
        "Kannada": "ಆಲೂಗಡ್ಡೆ",
        "Hindi": "आलू",
        "Telugu": "బంగాళాదుంప",
        "Tamil": "உருளைக்கிழங்கு"
    }[language]
})

        history.columns = [
            t["date"],
            t["crop"],
            t["disease"],
            t["confidence"],
            t["image"]
        ]

        st.dataframe(history)

        st.download_button(
            label=t["download_history"],
            data=history.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    else:

        st.info(t["no_history"])

elif page == "AI Assistant":

    st.title("🤖 Crop AI Assistant")
    if st.button("🗑 New Chat"):
       st.session_state.messages = []
       st.rerun()
    st.caption("💬 Ask me anything about the currently detected crop disease.")

    # ---------------- CHAT MEMORY ----------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------------- SHOW CURRENT PREDICTION ----------------

    if "current_disease" in st.session_state:

        st.success(f"""
🌿 Latest Prediction

🦠 Disease : {st.session_state['current_disease']}

🎯 Confidence : {st.session_state['current_confidence']:.2f}%

💚 Health Score : {st.session_state['current_health_score']}/100
""")

    else:

        st.warning("⚠ Please detect a crop disease first.")

    st.divider()

    # ---------------- DISPLAY CHAT ----------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # ---------------- USER INPUT ----------------

    prompt = st.chat_input("Ask anything about your crop...")

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):

            st.markdown(prompt)

        answer = get_ai_response(prompt)

        with st.chat_message("assistant"):

            st.markdown(answer)

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer

        })
        # ---------------- SAVE CHAT HISTORY ----------------

        chat = pd.DataFrame({
          "Time": [datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
          "Disease": [st.session_state.get("current_disease", "")],
          "Confidence": [st.session_state.get("current_confidence", 0)],
          "Question": [prompt],
          "Answer": [answer]
})

        if os.path.exists("chat_history.csv"):
            chat.to_csv("chat_history.csv", mode="a", header=False, index=False)
        else:
           chat.to_csv("chat_history.csv", index=False)

# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.markdown("""
    <div class="hero">

    <div class="hero-title">
    📘 About Project
    </div>

    <div class="hero-subtitle">
    AI-Powered Smart Agriculture System using Deep Learning and Computer Vision
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
# 🌱 AI-Powered Potato Crop Disease Detection System

This project is an Artificial Intelligence based smart agriculture application developed to detect potato leaf diseases using Deep Learning and Computer Vision technologies.

The system analyzes uploaded potato leaf images and predicts diseases with confidence score, severity level, health status, and treatment recommendations.

---

# 🎯 Main Objective

The main objective of this project is:

✅ Early crop disease detection  
✅ Reduce crop damage  
✅ Help farmers identify diseases quickly  
✅ Improve smart farming using AI  
✅ Provide multilingual farmer assistance  

---

# 🧠 Technologies Used

## 🔹 Frontend Technologies

- Streamlit
- HTML
- CSS
- Glassmorphism Dashboard UI

## 🔹 Backend Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas

## 🔹 AI & Deep Learning

- CNN (Convolutional Neural Network)
- Image Classification
- Computer Vision
- Deep Learning Algorithms

---

# 🌿 Supported Potato Diseases

## 🟠 Potato Early Blight

### Symptoms
- Brown circular spots
- Yellowing leaves
- Dry damaged areas

### Solutions
- Use fungicides
- Remove infected leaves
- Maintain proper irrigation

---

## 🔴 Potato Late Blight

### Symptoms
- Black wet lesions
- White fungal growth
- Rapid leaf decay

### Solutions
- Use Mancozeb fungicide
- Improve airflow
- Remove infected plants

---

## 🟤 Potato Common Scab

### Symptoms
- Rough potato surface
- Corky lesions
- Cracked potato skin

### Solutions
- Maintain soil moisture
- Use healthy seeds
- Rotate crops regularly

---

## 🟢 Healthy Potato Leaf

Indicates the crop is healthy and disease-free.

---

# ⚡ Key Features

✅ AI-Based Disease Detection  
✅ Real-Time Prediction  
✅ Disease Severity Detection  
✅ Smart Health Score System  
✅ Plant Health Status  
✅ AI Treatment Recommendations  
✅ Disease Probability Analysis  
✅ Downloadable AI Reports  
✅ Multilingual Support  
✅ Modern Futuristic Dashboard  

---

# 🌍 Multilingual Support

This system supports:

- English
- Kannada
- Hindi

This helps farmers understand disease information in their preferred language.

---

# 📊 Working Process

## Step 1
Upload potato leaf image.

## Step 2
Image preprocessing is performed using Computer Vision.

## Step 3
CNN Deep Learning model analyzes disease patterns.

## Step 4
AI predicts disease with confidence score.

## Step 5
System generates:
- Severity level
- Health score
- Health status
- Treatment recommendation

## Step 6
User can download AI report.

---

# 🧪 Dataset Used

The model was trained using potato leaf disease image datasets collected from:

- PlantVillage Dataset
- Kaggle Agricultural Datasets

The dataset contains:
- Healthy leaves
- Early Blight images
- Late Blight images
- Common Scab images

---

# 🤖 AI Model Details

## Model Type
Convolutional Neural Network (CNN)

## AI Functions
- Image Classification
- Disease Detection
- Feature Extraction
- Probability Prediction

## Training Process
- Image preprocessing
- Resizing
- Normalization
- CNN training
- Model evaluation

---

# 📈 Project Advantages

✅ Fast disease detection  
✅ Reduces farming losses  
✅ Improves crop monitoring  
✅ Easy-to-use interface  
✅ Supports smart agriculture  
✅ Helps beginner farmers  

---

# 🚀 Future Improvements

🔹 Live camera disease detection  
🔹 Mobile application support  
🔹 Weather-based prediction  
🔹 Cloud database integration  
🔹 More crop disease support  
🔹 Voice assistant integration  
🔹 IoT smart farming system  

---

# 👨‍💻 Project Outcome

This project demonstrates how Artificial Intelligence and Deep Learning can improve agriculture through smart crop disease detection systems.

The system combines:
- AI
- Computer Vision
- Deep Learning
- Multilingual Support
- Smart Dashboard Design

to create an intelligent farming assistant for modern agriculture.

---

# 🌱 Conclusion

AI-powered agriculture systems can help farmers detect diseases early, reduce crop damage, and improve productivity.

This project shows the future potential of Artificial Intelligence in smart farming and sustainable agriculture.
""")