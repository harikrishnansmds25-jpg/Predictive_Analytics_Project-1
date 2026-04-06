"""
Mental Health Status Classification - Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pickle
import re
import numpy as np

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stTextArea textarea { font-size: 15px; }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 1rem;
        font-size: 13px;
        color: #555;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# ── TEXT PREPROCESSING ───────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── LABEL COLORS & ICONS ─────────────────────────────────────────────────────
LABEL_CONFIG = {
    'Depression': {'color': '#6b4c9a', 'bg': '#f3eefb', 'icon': '🟣',
                   'desc': 'Text contains markers of depression — hopelessness, sadness, or loss of interest.'},
    'Anxiety':    {'color': '#c0572a', 'bg': '#fdf0eb', 'icon': '🟠',
                   'desc': 'Text contains patterns of anxiety — worry, panic, or persistent fear.'},
    'PTSD':       {'color': '#8a6c2a', 'bg': '#faf5e8', 'icon': '🟡',
                   'desc': 'Text contains PTSD signals — trauma references, hypervigilance, or flashbacks.'},
    'Normal':     {'color': '#2a7a4a', 'bg': '#eaf7ef', 'icon': '🟢',
                   'desc': 'No significant mental health risk indicators detected.'},
}

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("🧠 Mental Health Status Classifier")
st.markdown("**Classifies social media text into:** Depression · Anxiety · PTSD · Normal")
st.markdown("---")

# ── SAMPLE TEXTS ─────────────────────────────────────────────────────────────
st.markdown("#### Try a sample or enter your own text")

samples = {
    "Select a sample...": "",
    "😔 Depression": "I feel completely empty inside. Nothing brings me joy anymore and I don't see the point of getting out of bed.",
    "😰 Anxiety": "My heart is racing and I can't stop worrying about everything that could go wrong. I feel like something terrible is about to happen.",
    "😨 PTSD": "I keep having flashbacks of that traumatic experience. I can't go near that place without feeling triggered and panicked.",
    "😊 Normal": "Had a great day today! Went for a walk, met some friends, and feeling really content and grateful."
}

selected = st.selectbox("Sample texts", list(samples.keys()))

# ── TEXT INPUT ───────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Enter social media post text:",
    value=samples[selected],
    height=150,
    placeholder="Type or paste a social media post here..."
)

# ── PREDICT ──────────────────────────────────────────────────────────────────
if st.button("🔍 Classify Text", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            model = load_model()
        except FileNotFoundError:
            st.error("❌ model.pkl not found! Run `python train_model.py` first.")
            st.stop()

        with st.spinner("Analysing..."):
            cleaned = clean_text(user_input)
            prediction = model.predict([cleaned])[0]

            # Get probabilities if available
            has_proba = hasattr(model, 'predict_proba')
            if has_proba:
                proba = model.predict_proba([cleaned])[0]
                classes = model.classes_
            else:
                # LinearSVC uses decision function
                decision = model.decision_function([cleaned])[0]
                proba = np.exp(decision) / np.sum(np.exp(decision))
                classes = model.classes_

        cfg = LABEL_CONFIG.get(prediction, LABEL_CONFIG['Normal'])

        # ── RESULT ───────────────────────────────────────────────────────
        st.markdown("### Result")
        st.markdown(f"""
        <div class="result-box" style="background:{cfg['bg']};border-color:{cfg['color']}">
            <h2 style="color:{cfg['color']};margin:0">{cfg['icon']} {prediction}</h2>
            <p style="margin:0.5rem 0 0;color:#444">{cfg['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE BARS ───────────────────────────────────────────────
        st.markdown("#### Confidence Scores")
        prob_dict = dict(zip(classes, proba))

        for label in ['Depression', 'Anxiety', 'PTSD', 'Normal']:
            if label in prob_dict:
                pct = prob_dict[label]
                color = LABEL_CONFIG[label]['color']
                st.markdown(f"**{LABEL_CONFIG[label]['icon']} {label}**")
                st.progress(float(pct))
                st.caption(f"{pct*100:.1f}%")

        # ── TOP PREDICTION CONFIDENCE ─────────────────────────────────────
        top_conf = prob_dict.get(prediction, 0)
        st.info(f"🎯 Prediction confidence: **{top_conf*100:.1f}%**")

# ── ABOUT SECTION ────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this project"):
    st.markdown("""
    **Project:** Mental Health Status Classification from Social Media Text  
    **Course:** Predictive Analytics — Academic Year 2025-26  
    **Team:** Harikrishnan S · Krithika S · Umapravathy C S  

    **Methodology:**
    - Text preprocessing: lowercasing, URL/mention removal, tokenization
    - Feature extraction: TF-IDF (unigrams + bigrams, 50k features)
    - Models trained: Logistic Regression, Linear SVM, Random Forest
    - Best model selected by accuracy on 20% held-out test set

    **Dataset:** Reddit mental health posts (Kaggle)  
    **Classes:** Depression · Anxiety · PTSD · Normal
    """)

# ── DISCLAIMER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This tool is for academic and research purposes only. 
It is NOT a substitute for professional mental health diagnosis or treatment. 
If you or someone you know is struggling, please contact a qualified mental health professional 
or call a crisis helpline.
</div>
""", unsafe_allow_html=True)
