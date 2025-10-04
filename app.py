import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import time

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="🍎 AI Produce Classifier",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ULTRA beautiful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Card styling with glassmorphism */
    .main-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.4);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 1rem 0;
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Title styling with glow effect */
    .title-text {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-shadow: 0 0 30px rgba(255,255,255,0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(255,255,255,0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(255,255,255,0.8)); }
    }
    
    .subtitle-text {
        text-align: center;
        color: #ffffff;
        font-size: 1.4rem;
        margin-bottom: 3rem;
        font-weight: 300;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        animation: fadeIn 1.2s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Upload box with hover effect */
    .uploadBox {
        border: 3px dashed rgba(255,255,255,0.6);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .uploadBox:hover {
        border-color: rgba(255,255,255,1);
        background: rgba(255,255,255,0.2);
        transform: scale(1.03);
        box-shadow: 0 15px 40px rgba(255,255,255,0.2);
    }
    
    /* Result card with 3D effect */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.15) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(255,255,255,0.4);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        margin: 1rem 0;
        animation: scaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        transform-style: preserve-3d;
    }
    
    @keyframes scaleIn {
        from { transform: scale(0.5) rotateY(20deg); opacity: 0; }
        to { transform: scale(1) rotateY(0deg); opacity: 1; }
    }
    
    .result-emoji {
        font-size: 6rem;
        animation: bounce 1.5s ease infinite;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.3));
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-20px) scale(1.1); }
    }
    
    .result-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 1.5rem 0;
        text-shadow: 0 3px 10px rgba(0,0,0,0.4);
        letter-spacing: 1px;
    }
    
    .confidence-text {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 50%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 3px 10px rgba(255,215,0,0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Confidence bar with gradient */
    .confidence-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 50px;
        height: 50px;
        margin: 2rem 0;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
    }
    
    .confidence-bar {
        height: 100%;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 50%, #4facfe 100%);
        background-size: 200% 200%;
        border-radius: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        color: white;
        text-shadow: 0 2px 5px rgba(0,0,0,0.3);
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        animation: shimmer 3s ease infinite;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.6);
    }
    
    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Prediction items with hover animation */
    .prediction-item {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1.3rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .prediction-item:nth-child(1) { border-color: #ffd700; }
    .prediction-item:nth-child(2) { border-color: #c0c0c0; }
    .prediction-item:nth-child(3) { border-color: #cd7f32; }
    .prediction-item:nth-child(4) { border-color: #667eea; }
    .prediction-item:nth-child(5) { border-color: #764ba2; }
    
    .prediction-item:hover {
        transform: translateX(10px) scale(1.02);
        background: rgba(255, 255, 255, 0.25);
        box-shadow: 0 10px 30px rgba(255,255,255,0.3);
    }
    
    .pred-name {
        font-size: 1.3rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
        color: white;
        text-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    .pred-confidence {
        font-size: 1.5rem;
        font-weight: 900;
        color: #ffd700;
        text-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.3);
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .info-box h3 {
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    
    .info-box ul {
        list-style: none;
        padding: 0;
    }
    
    .info-box li {
        padding: 0.7rem 0;
        font-size: 1.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }
    
    .info-box li:last-child {
        border-bottom: none;
    }
    
    /* Button styling with gradient */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        transition: all 0.4s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Progress bar animation */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        animation: progressShine 1s ease infinite;
    }
    
    @keyframes progressShine {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: white;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Image styling */
    img {
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        transition: transform 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Emoji mapping
EMOJI_MAP = {
    'apple': '🍎', 'banana': '🍌', 'beetroot': '🥬', 'bell pepper': '🫑',
    'cabbage': '🥬', 'capsicum': '🫑', 'carrot': '🥕', 'cauliflower': '🥦',
    'chilli pepper': '🌶️', 'corn': '🌽', 'cucumber': '🥒', 'eggplant': '🍆',
    'garlic': '🧄', 'ginger': '🫚', 'grapes': '🍇', 'jalepeno': '🌶️',
    'kiwi': '🥝', 'lemon': '🍋', 'lettuce': '🥬', 'mango': '🥭',
    'onion': '🧅', 'orange': '🍊', 'paprika': '🫑', 'pear': '🍐',
    'peas': '🫛', 'pineapple': '🍍', 'pomegranate': '🍒', 'potato': '🥔',
    'radish': '🌰', 'soy beans': '🫘', 'spinach': '🥬', 'sweetcorn': '🌽',
    'sweetpotato': '🍠', 'tomato': '🍅', 'turnip': '🌰', 'watermelon': '🍉'
}

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
        'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
        'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
        'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
        'pomegranate', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn',
        'sweetpotato', 'tomato', 'turnip', 'watermelon'
    ]

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess image
def preprocess_image(image, target_size=(180, 180)):
    img = image.resize(target_size)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Function to predict
def predict_image(model, image, class_names):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0]).numpy()  # Convert to numpy
    
    predicted_class = class_names[np.argmax(score)]
    confidence = float(100 * np.max(score))  # Convert to float
    
    # Get top 5 predictions
    top_5_idx = np.argsort(score)[-5:][::-1]
    top_5_classes = [class_names[i] for i in top_5_idx]
    top_5_confidences = [float(100 * score[i]) for i in top_5_idx]  # Convert to float
    
    return predicted_class, confidence, top_5_classes, top_5_confidences

# ============= MAIN APP =============

# Header
st.markdown('<p class="title-text">🥗 AI PRODUCE CLASSIFIER</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">✨ Identify fruits & vegetables instantly with AI-powered computer vision ✨</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    model_path = st.text_input(
        "🔧 Model Path",
        value="Image_Classification_Model.keras",
        help="Enter the path to your trained model file"
    )
    
    if st.button("🚀 LOAD MODEL", use_container_width=True):
        with st.spinner("🔄 Loading model..."):
            st.session_state.model = load_model(model_path)
            if st.session_state.model:
                st.success("✅ Model loaded successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to load model")
    
    st.markdown("---")
    
    st.markdown("## 📊 System Status")
    if st.session_state.model:
        st.success("**🟢 ONLINE**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📦 Classes", len(st.session_state.class_names))
        with col2:
            st.metric("📐 Input", "180×180")
    else:
        st.error("**🔴 OFFLINE**")
        st.warning("⚠️ Please load model to start")
    
    st.markdown("---")
    
    st.markdown("## 📖 Quick Guide")
    st.markdown("""
    **1️⃣** Load the AI model  
    **2️⃣** Upload fruit/veggie image  
    **3️⃣** Click analyze button  
    **4️⃣** Get instant predictions!
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 36 Categories")
    with st.expander("📋 View All"):
        for item in st.session_state.class_names:
            emoji = EMOJI_MAP.get(item, '🥗')
            st.write(f"{emoji} **{item.title()}**")

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Image")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a fruit or vegetable",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='📸 Your Uploaded Image', use_container_width=True)
        
        # Predict button
        if st.button("🔮 ANALYZE IMAGE", use_container_width=True, type="primary"):
            if st.session_state.model is None:
                st.error("⚠️ Please load the model first from the sidebar!")
            else:
                with st.spinner("🔍 AI is analyzing your image..."):
                    # Add progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.008)
                        progress_bar.progress(i + 1)
                    
                    predicted_class, confidence, top_5_classes, top_5_confidences = predict_image(
                        st.session_state.model,
                        image,
                        st.session_state.class_names
                    )
                    
                    # Store results
                    st.session_state.predicted_class = predicted_class
                    st.session_state.confidence = confidence
                    st.session_state.top_5_classes = top_5_classes
                    st.session_state.top_5_confidences = top_5_confidences
                    
                    progress_bar.empty()
                    st.success("✅ Analysis Complete!")
                    st.balloons()
                    time.sleep(0.3)
                    st.rerun()
    else:
        st.markdown('<div class="uploadBox">', unsafe_allow_html=True)
        st.markdown("### 📁 DROP IMAGE HERE")
        st.markdown("**Supported:** JPG, JPEG, PNG")
        st.markdown("**Max Size:** 16MB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if hasattr(st.session_state, 'predicted_class'):
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        
        # Main result card
        emoji = EMOJI_MAP.get(st.session_state.predicted_class, '🥗')
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-emoji">{emoji}</div>
            <div class="result-title">{st.session_state.predicted_class.upper()}</div>
            <div class="confidence-text">{st.session_state.confidence:.1f}%</div>
            <div class="confidence-container">
                <div class="confidence-bar" style="width: {st.session_state.confidence}%">
                    CONFIDENCE: {st.session_state.confidence:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics row
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("🎯 Predicted", st.session_state.predicted_class.title())
        with col_b:
            st.metric("📊 Score", f"{st.session_state.confidence:.1f}%")
        with col_c:
            if st.session_state.confidence > 85:
                status = "✅ HIGH"
            elif st.session_state.confidence > 70:
                status = "⚡ GOOD"
            else:
                status = "⚠️ LOW"
            st.metric("💪 Quality", status)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Top 5 predictions
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 TOP 5 PREDICTIONS")
        
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        
        for i, (cls, conf) in enumerate(zip(st.session_state.top_5_classes, st.session_state.top_5_confidences)):
            emoji = EMOJI_MAP.get(cls, '🥗')
            
            st.markdown(f"""
            <div class="prediction-item">
                <div class="pred-name">
                    <span style="font-size: 1.8rem;">{medals[i]}</span>
                    <span style="font-size: 1.5rem;">{emoji}</span>
                    <span>{cls.upper()}</span>
                </div>
                <div class="pred-confidence">{conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Colored progress bar for each
            st.progress(conf / 100)
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 RESULTS PANEL")
        
        st.markdown("""
        <div class="info-box">
            <h3>💡 PRO TIPS FOR BEST RESULTS</h3>
            <ul>
                <li>✅ Use high-quality, clear images</li>
                <li>✅ Ensure good lighting conditions</li>
                <li>✅ Center the item in the frame</li>
                <li>✅ Avoid cluttered backgrounds</li>
                <li>✅ Single item works best</li>
                <li>✅ Close-up shots are preferred</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("🚀 **Ready to classify!** Upload an image to see AI predictions with confidence scores!")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem;'>
    <h2 style='font-weight: 700; margin-bottom: 1rem;'>⚡ POWERED BY DEEP LEARNING</h2>
    <p style='font-size: 1.2rem; opacity: 0.9;'>Built with TensorFlow, Keras & Streamlit</p>
    <p style='font-size: 1rem; opacity: 0.7; margin-top: 1rem;'>
        🧠 CNN Architecture | 📦 36 Categories | 🎯 High Accuracy
    </p>
</div>
""", unsafe_allow_html=True)