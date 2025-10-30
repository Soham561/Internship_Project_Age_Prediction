import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
from datetime import datetime
import pandas as pd
import plotly.express as px

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="AgeSense AI - Advanced Age Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS -----------------
def inject_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
    .confidence-bar {
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, #ff4b4b, #ffa500, #ffff00, #c0ff00, #00ff00);
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    """Load the face detection and age prediction models"""
    try:
        # Update these paths or provide a way to configure them
        face_proto = r"S:\Age-prediction\opencv_face_detector.pbtxt"
        face_model = r"S:\Age-prediction\opencv_face_detector_uint8.pb"
        age_proto = r"S:\Age-prediction\age_deploy.prototxt"
        age_model = r"S:\Age-prediction\age_net.caffemodel"
        
        # Check if model files exist
        for path in [face_proto, face_model, age_proto, age_model]:
            if not os.path.exists(path):
                st.error(f"Model file not found: {path}")
                st.info("Please ensure all model files are in the 'models' directory")
                return None, None
        
        # Load networks
        face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
        age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        
        return face_net, age_net
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Age groups
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Mean values for model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load models
face_net, age_net = load_models()

# ----------------- Helper Functions -----------------
def detect_faces_and_predict_age(image, confidence_threshold=0.7):
    """Detect faces and predict ages for an input PIL image."""
    if face_net is None or age_net is None:
        st.error("Models not loaded. Please check the model files.")
        return [], image
    
    img_cv = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    blob = cv2.dnn.blobFromImage(img_cv, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = img_cv[y1:y2, x1:x2]

            if face.size == 0:
                continue

            blob_face = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            age_net.setInput(blob_face)
            age_preds = age_net.forward()
            age_idx = age_preds[0].argmax()
            age = AGE_BUCKETS[age_idx]
            age_confidence = age_preds[0][age_idx]

            # Draw bounding box and label
            label = f"{age} ({age_confidence*100:.1f}%)"
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            results.append({
                "age": age,
                "age_confidence": age_confidence,
                "face_confidence": confidence,
                "box": (x1, y1, x2, y2),
                "cropped_face": face
            })

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return results, Image.fromarray(img_rgb)

# ----------------- Login Page -----------------
def login():
    st.markdown("<h1 class='main-header'>AgeSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Advanced Facial Age Prediction System</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("<h2 style='text-align: center;'>🔐 Secure Login</h2>", unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.button("Login", use_container_width=True):
                if username == "admin" and password == "1234":
                    st.session_state["authenticated"] = True
                    st.session_state["login_time"] = datetime.now()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            
            st.markdown("<div style='text-align: center; margin-top: 2rem;'>"
                       "Demo Credentials: admin / 1234</div>", unsafe_allow_html=True)

# ----------------- Main App -----------------
def app():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/calendar.png", width=80)
        st.markdown("<h2>AgeSense AI</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Settings
        st.markdown("<h3>⚙️ Settings</h3>", unsafe_allow_html=True)
        confidence_threshold = st.slider("Face Detection Confidence", 0.1, 1.0, 0.7, 0.05)
        
        # Model info
        st.markdown("---")
        st.markdown("<h3>🧠 Model Information</h3>", unsafe_allow_html=True)
        if face_net is not None and age_net is not None:
            st.success("Primary models loaded successfully")
        
        # Session info
        st.markdown("---")
        if "login_time" in st.session_state:
            login_time = st.session_state["login_time"]
            st.markdown(f"**Login Time:** {login_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    st.markdown("<h1 class='main-header'>AgeSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Advanced Facial Age Prediction System</p>", 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload/Capture", "📊 Analysis Results", "📈 History & Analytics"])
    
    # ---- TAB 1: Upload/Capture ----
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3 class='sub-header'>Image Input</h3>", unsafe_allow_html=True)
            option = st.radio("Choose Input Method:", 
                             ["📤 Upload Image", "📸 Take Photo"], 
                             horizontal=True)
            image = None
            
            if option == "📤 Upload Image":
                uploaded_file = st.file_uploader("Choose an image...",
                                                type=["jpg", "jpeg", "png"],
                                                help="Upload a clear frontal face image for best results")
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.session_state["uploaded_file_name"] = uploaded_file.name
            
            elif option == "📸 Take Photo":
                camera_file = st.camera_input("Take a picture for age analysis")
                if camera_file is not None:
                    image = Image.open(camera_file)
                    st.session_state["uploaded_file_name"] = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            if image is not None:
                # Store original image
                st.session_state["original_image"] = image
                
                # Display image info
                st.info(f"Image format: {image.format if hasattr(image, 'format') else 'Unknown'}, "
                       f"Size: {image.size[0]}x{image.size[1]} pixels, "
                       f"Mode: {image.mode}")
        
        with col2:
            st.markdown("<h3 class='sub-header'>Analysis Options</h3>", unsafe_allow_html=True)
            
            # Add analysis options
            enhance_quality = st.checkbox("Enhance image quality", value=True)
            show_confidence = st.checkbox("Show confidence scores", value=True)
            
            if st.button("Analyze Image", type="primary", use_container_width=True):
                if "original_image" in st.session_state:
                    with st.spinner("🔍 Analyzing image... Please wait..."):
                        progress_bar = st.progress(0)
                        
                        for i in range(100):
                            time.sleep(0.02)  # Simulate processing
                            progress_bar.progress(i + 1)
                        
                        results, annotated_img = detect_faces_and_predict_age(
                            st.session_state["original_image"], confidence_threshold)
                        
                        # Store results in session state
                        st.session_state["results"] = results
                        st.session_state["annotated_img"] = annotated_img
                        st.session_state["analysis_time"] = datetime.now()
                        
                        # Add to history
                        if "history" not in st.session_state:
                            st.session_state["history"] = []
                        
                        st.session_state["history"].append({
                            "timestamp": st.session_state["analysis_time"],
                            "filename": st.session_state.get("uploaded_file_name", "unknown"),
                            "faces_detected": len(results),
                            "results": results
                        })
                    
                    st.success("Analysis completed!")
                    st.balloons()
                else:
                    st.warning("Please upload or capture an image first.")
    
    # ---- TAB 2: Analysis Results ----
    with tab2:
        if "results" in st.session_state and "annotated_img" in st.session_state:
            st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state["annotated_img"],
                         caption="Detected Faces with Age Predictions",
                         use_container_width=True)
                
                # Add download button for annotated image
                annotated_img_bytes = cv2.imencode('.jpg', 
                                                  np.array(st.session_state["annotated_img"].convert('RGB')))[1].tobytes()
                st.download_button(
                    label="Download Annotated Image",
                    data=annotated_img_bytes,
                    file_name=f"annotated_{st.session_state.get('uploaded_file_name', 'image')}",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            with col2:
                st.markdown(f"**Analysis Time:** {st.session_state['analysis_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Faces Detected:** {len(st.session_state['results'])}")
                
                for idx, res in enumerate(st.session_state["results"]):
                    with st.container():
                        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                        st.markdown(f"**Face {idx+1}**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(res["cropped_face"], 
                                    caption=f"Face {idx+1}",
                                    use_container_width=True)
                        with col_b:
                            st.markdown(f"**Predicted Age:** {res['age']}")
                            
                            # Confidence visualization
                            st.markdown(f"**Age Confidence:** {res['age_confidence']*100:.1f}%")
                            st.markdown(f"""
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {res['age_confidence']*100}%;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"**Face Detection Confidence:** {res['face_confidence']*100:.1f}%")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ Please upload an image and analyze it first to see results.")
    
    # ---- TAB 3: History & Analytics ----
    with tab3:
        st.markdown("<h3 class='sub-header'>Analysis History</h3>", unsafe_allow_html=True)
        
        if "history" in st.session_state and st.session_state["history"]:
            # Create dataframe from history
            history_data = []
            for record in st.session_state["history"]:
                for face in record["results"]:
                    history_data.append({
                        "Timestamp": record["timestamp"],
                        "Filename": record["filename"],
                        "Age": face["age"],
                        "Age Confidence": face["age_confidence"],
                        "Face Confidence": face["face_confidence"]
                    })
            
            df = pd.DataFrame(history_data)
            
            if not df.empty:
                # Show data table
                st.dataframe(df, use_container_width=True)
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                    st.markdown(f"**Total Analyses:** {len(st.session_state['history'])}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                    st.markdown(f"**Total Faces Detected:** {len(df)}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                    if not df.empty:
                        most_common_age = df['Age'].mode()[0]
                        st.markdown(f"**Most Common Age:** {most_common_age}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                    if not df.empty:
                        avg_confidence = df['Age Confidence'].mean() * 100
                        st.markdown(f"**Avg Confidence:** {avg_confidence:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Age distribution chart
                st.markdown("<h4>Age Distribution</h4>", unsafe_allow_html=True)
                age_counts = df['Age'].value_counts().reset_index()
                age_counts.columns = ['Age', 'Count']
                
                fig = px.bar(age_counts, x='Age', y='Count', 
                            title="Distribution of Predicted Ages",
                            color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                st.markdown("<h4>Confidence Distribution</h4>", unsafe_allow_html=True)
                fig2 = px.histogram(df, x='Age Confidence', 
                                   title="Distribution of Age Confidence Scores",
                                   nbins=20, color_discrete_sequence=['#ff7f0e'])
                st.plotly_chart(fig2, use_container_width=True)
            
            # Clear history button
            if st.button("Clear History", type="secondary"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.info("No analysis history available. Analyze some images to build history.")
    
    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>AgeSense AI • Advanced Facial Age Prediction • v2.0</div>", 
                unsafe_allow_html=True)

# ----------------- Run App -----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    if face_net is not None and age_net is not None:
        app()
    else:
        st.error("Unable to load required models. Please check the model files and try again.")
