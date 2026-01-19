import streamlit as st
import joblib
import re
import string

# Page Configuration
st.set_page_config(page_title="TruthGuard AI", page_icon="🛡️", layout="wide")

# Load Model & Vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_model()
except:
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

# Text Cleaning Function (Must match training logic)
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# --- UI DESIGN ---
st.markdown("""
    <style>
    .main_header {text-align: center; color: #4A90E2;}
    .stTextArea textarea {font-size: 16px;}
    .real-news {padding: 20px; border-radius: 10px; background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .fake-news {padding: 20px; border-radius: 10px; background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main_header'>🛡️ TruthGuard: Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("### 🔍 Verify news credibility instantly using AI")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Paste the news article or headline here:", height=250, placeholder="Example: Breaking news about...")
    
    if st.button("Analyze News 🚀", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Prediction Logic
            cleaned_text = wordopt(user_input)
            vec_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vec_text)
            
            # Display Result
            st.write("---")
            if prediction[0] == 1:
                st.markdown(f"""
                <div class='real-news'>
                    <h2>✅ Likely REAL News</h2>
                    <p>Our AI analysis suggests this content comes from credible patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='fake-news'>
                    <h2>🚨 Likely FAKE News</h2>
                    <p>Warning: This content matches patterns often found in misinformation.</p>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.info("ℹ️ **How it works?**")
    st.write("This AI uses a **Passive Aggressive Classifier** trained on over 40,000 articles.")
    st.markdown("**Dataset Stats:**")
    st.caption("- fake.csv: Conspiracy & Clickbait")
    st.caption("- true.csv: Reuters & Verified Sources")
    
    st.divider()

    st.write("please provide ***full  detailed correct lengthy news in paragraph formaat to better understand for ai*** to check whether it is fake or real news")
