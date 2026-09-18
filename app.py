from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np

from src.config import ARTIFACT_DIR, LABELS
from src.preprocessing import preprocess
from src.predict import load_model

st.set_page_config(
    page_title="Bangla Toxic & Cyberbullying Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a modern, sleek appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Bangla Toxic Comment & Cyberbullying Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">বাংলা ও বাংলিশ (Banglish) টেক্সটের ক্ষতিকর মন্তব্য ও সাইবারবুলিং শনাক্তকরণ সিস্টেম</div>', unsafe_allow_html=True)

models = sorted(p.stem for p in ARTIFACT_DIR.glob("*.joblib"))
if not models:
    st.warning("⚠️ No trained model found in artifacts/. Please train a model first.")
    st.code("python -m src.data_prep\npython -m src.train --model tfidf_lr")
    st.stop()

with st.sidebar:
    st.header("⚙️ মডেল সেটিংস (Settings)")
    model_name = st.selectbox("মডেল নির্বাচন করুন (Choose model):", models)
    
    metrics_path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    report = {}
    best_thresholds = {lbl: 0.5 for lbl in LABELS}
    if metrics_path.exists():
        try:
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            raw_th = report.get("best_thresholds", {})
            if isinstance(raw_th, dict):
                for lbl in LABELS:
                    if lbl in raw_th:
                        best_thresholds[lbl] = float(raw_th[lbl])
        except Exception:
            pass

    threshold_mode = st.radio(
        "থ্রেশহোল্ড মোড (Threshold Mode):",
        ["🎯 মডেলের অপ্টিমাইজড থ্রেশহোল্ড (Recommended)", "🎚️ কাস্টম থ্রেশহোল্ড স্লাইডার"]
    )
    
    if threshold_mode == "🎚️ কাস্টম থ্রেশহোল্ড স্লাইডার":
        custom_th = float(st.slider("লেবেল থ্রেশহোল্ড (Detection Threshold):", 0.05, 0.95, 0.50, 0.05))
        active_thresholds = {lbl: custom_th for lbl in LABELS}
    else:
        active_thresholds = best_thresholds

    st.markdown("---")
    st.subheader("📊 মডেল পারফরম্যান্স")
    if report:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Test Micro-F1", f"{report.get('test', {}).get('micro_f1', 0.0):.3f}")
        with col_m2:
            st.metric("Test Macro-F1", f"{report.get('test', {}).get('macro_f1', 0.0):.3f}")
        
        with st.expander("🔍 বর্তমান সক্রিয় থ্রেশহোল্ড"):
            for lbl, th in active_thresholds.items():
                st.write(f"• **{lbl}**: `{th:.2f}` ({th*100:.0f}%)")

    st.markdown("---")
    st.caption("Course: NLP Multi-Label Toxic Classifier • From-Scratch ML")

# Example Sentences for Quick Testing
SAMPLE_SENTENCES = {
    "✨ পছন্দ করুন (Select an example)": "",
    "🟢 [সাধারণ প্রশ্ন] এই বইটা কোথায় কিনতে পাওয়া যাবে?": "এই বইটা কোথায় কিনতে পাওয়া যাবে?",
    "🟢 [পজিটিভ/নিউট্রাল] আজকের আবহাওয়াটা অনেক সুন্দর এবং চমৎকার।": "আজকের আবহাওয়াটা অনেক সুন্দর এবং চমৎকার।",
    "🟢 [প্রশংসা] ভাই আপনার কাজটি সত্যিই অনেক তথ্যবহুল ও দারুণ ছিল!": "ভাই আপনার কাজটি সত্যিই অনেক তথ্যবহুল ও দারুণ ছিল!",
    "🔴 [বুলিং/অশালীন] তুই একটা আস্ত কুত্তার বাচ্চা তোরে জুতা মারমু": "তুই একটা আস্ত কুত্তার বাচ্চা তোরে জুতা মারমু",
    "🔴 [হুমকি/Threat] তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু, জানে শেষ করে দেব": "তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু, জানে শেষ করে দেব",
    "🔴 [বিদ্বেষ/Hate Speech] এই ফকিন্নি মালাউনদের দেশ থেকে লাথি মেরে তাড়ানো উচিত": "এই ফকিন্নি মালাউনদের দেশ থেকে লাথি মেরে তাড়ানো উচিত",
    "🟡 [বাংলিশ পজিটিভ] video ta onek sundor hoyeche bro, shuvokamona roilo": "video ta onek sundor hoyeche bro, shuvokamona roilo",
    "🔴 [বাংলিশ ক্ষতিকর] tui ekta baje faltu manush, tore dekhle shobai ghenna kore": "tui ekta baje faltu manush, tore dekhle shobai ghenna kore"
}

# Managing session state for text input
if "user_comment" not in st.session_state:
    st.session_state.user_comment = "এই বইটা কোথায় কিনতে পাওয়া যাবে?"

def on_sample_change():
    chosen = st.session_state.sample_selector
    if chosen and SAMPLE_SENTENCES.get(chosen):
        st.session_state.user_comment = SAMPLE_SENTENCES[chosen]

st.markdown("##### 🧪 টেস্ট করার জন্য নমুনা বাক্য (Sample Test Sentences):")
st.selectbox(
    "ক্লিক করে উদাহরণ সিলেক্ট করুন:",
    options=list(SAMPLE_SENTENCES.keys()),
    index=0,
    key="sample_selector",
    on_change=on_sample_change,
    label_visibility="collapsed"
)

# Text area
text = st.text_area(
    "বাংলা / Banglish comment লিখুন বা উপরের উদাহরণ থেকে বেছে নিন:",
    key="user_comment",
    height=120,
    placeholder="এখানে আপনার টেক্সট লিখুন..."
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    classify_clicked = st.button("🔍 Classify Text", type="primary", width="stretch")

def classify_comment(model, comment_text: str, thresholds_dict: dict):
    probs = np.asarray(model.predict_proba([comment_text]))[0]
    detected_labels = []
    for i, lbl in enumerate(LABELS):
        cutoff = float(thresholds_dict.get(lbl, 0.5))
        if float(probs[i]) >= cutoff:
            detected_labels.append(lbl)
    return {
        "labels": detected_labels,
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "neutral": len(detected_labels) == 0,
        "tokens": preprocess(comment_text),
    }

if classify_clicked or text.strip():
    if not text.strip():
        st.warning("⚠️ অনুগ্রহ করে কোনো টেক্সট বা কমেন্ট লিখুন।")
    else:
        model = load_model(model_name)
        result = classify_comment(model, text, active_thresholds)
        
        st.markdown("---")
        
        # Overall Verdict Banner
        if result["neutral"]:
            st.success("✅ **নিরাপদ ও নিউট্রাল মন্তব্য (Neutral / Safe)** — কোনো ক্ষতিকর লেবেল থ্রেশহোল্ড অতিক্রম করেনি।")
        else:
            detected_badges = " ".join([f"`{lbl.upper()}`" for lbl in result["labels"]])
            st.error(f"🚨 **ক্ষতিকর উপাদান শনাক্ত হয়েছে (Toxic Detected):** {detected_badges}")
        
        st.markdown("### 📊 প্রতিটি লেবেলের সম্ভাবনা (Label Probabilities & Percentages)")
        
        # Build tabular data with percentages and probabilities
        rows = []
        label_bn_map = {
            "hate_speech": "ঘৃণামূলক বক্তব্য (Hate Speech)",
            "sexist": "লিঙ্গবৈষম্যমূলক/অশালীন (Sexist)",
            "threat": "হুমকি (Threat)",
            "bullying": "সাইবারবুলিং (Bullying)",
            "toxic": "বিষাক্ত/ক্ষতিকর (Toxic)"
        }
        
        for lbl in LABELS:
            prob = result["probabilities"][lbl]
            pct = prob * 100.0
            th = float(active_thresholds.get(lbl, 0.5))
            is_detected = prob >= th
            status = "🚨 শনাক্ত (Detected)" if is_detected else "✅ নিরাপদ (Normal)"
            rows.append({
                "লেবেল (Label)": label_bn_map.get(lbl, lbl),
                "শতাংশ (Percentage)": f"{pct:.1f}%",
                "প্রবাবিলিটি (Probability)": round(prob, 4),
                "কাট-অফ (Threshold)": f"{th*100:.0f}%",
                "Confidence Meter": prob,
                "স্ট্যাটাস (Status)": status
            })
            
        df_display = pd.DataFrame(rows)
        
        # Display rich table with progress bars
        st.dataframe(
            df_display,
            column_config={
                "Confidence Meter": st.column_config.ProgressColumn(
                    "কনফিডেন্স মিটার (%)",
                    help="লেবেলের সম্ভাব্যতা শতকরা হিসেবে",
                    format="%.1f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "শতাংশ (Percentage)": st.column_config.TextColumn(
                    "শতাংশ (%)",
                    width="small"
                ),
                "প্রবাবিলিটি (Probability)": st.column_config.NumberColumn(
                    "প্রবাবিলিটি (0 - 1)",
                    format="%.4f",
                    width="small"
                ),
                "কাট-অফ (Threshold)": st.column_config.TextColumn(
                    "কাট-অফ",
                    width="small"
                ),
                "স্ট্যাটাস (Status)": st.column_config.TextColumn(
                    "স্ট্যাটাস",
                    width="medium"
                ),
            },
            hide_index=True,
            width="stretch"
        )
        
        # Token inspection accordion
        with st.expander("🔎 প্রি-প্রসেসড টোকেনসমূহ (Inspected Tokens)"):
            st.caption("মডেলের ক্লিনিং ও স্টপওয়ার্ড রিমুভালের পর শব্দগুলোর রূপ:")
            tokens = result["tokens"]
            if tokens:
                st.write(" | ".join([f"`{t}`" for t in tokens]))
            else:
                st.write("*(কোনো তাৎপর্যপূর্ণ টোকেন পাওয়া যায়নি)*")
