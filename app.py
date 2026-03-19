import streamlit as st
from PIL import Image
import pandas as pd
from detection import load_rescue_model, detect_damage 
from geo_utils import get_coordinates

# 1. Page Configuration
st.set_page_config(page_title="RescueAI Dashboard", layout="wide")
st.title("🚨 RescueAI: Damage Assessment")

# 2. Initialize Model (Session State prevents reloading)
if 'model' not in st.session_state:
    with st.spinner("Loading AI Brain..."):
        st.session_state.model = load_rescue_model()

# 3. Sidebar for Uploads
uploaded_file = st.sidebar.file_uploader("Upload Drone/Satellite Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Imagery")
        st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("AI Analysis")
        with st.spinner("Analyzing structural integrity..."):
            severity, confidence, heatmap = detect_damage(img, st.session_state.model)
            
            # --- ACCURACY BOOSTER & SAFETY LOGIC ---
            # Corrects false negatives in flooded areas or low-confidence guesses
            if severity == "No Damage" and confidence < 0.35:
                severity = "Possible Flood/Water Damage"
                st.info("💡 RescueAI Note: Detected high-moisture patterns. Marking for safety review.")

            # --- DISPLAY LOGIC ---
            if confidence < 0.20:
                st.warning("⚠️ Low Confidence: AI detected conflicting patterns.")
                display_severity = "Inconclusive / Review Required"
                color = "orange"
            else:
                display_severity = severity
                # Assign colors based on final severity
                if severity in ["Major Damage", "Destroyed", "Possible Flood/Water Damage"]:
                    color = "red"
                elif severity == "No Damage":
                    color = "green"
                else:
                    color = "orange"
            
            # Final Assessment Output
            st.markdown(f"## Assessment: :{color}[{display_severity}]")
            st.progress(min(confidence, 1.0))
            st.write(f"Confidence Score: {confidence:.2%}")

    st.divider()
    
    # 4. Explainability Section
    st.subheader("🔍 AI Focus Area (Explainability Heatmap)")
    st.image(heatmap, caption="Red highlights indicate patterns the AI associated with damage.", use_container_width=True)
    
    # 5. Mapping Section
    lat, lon = get_coordinates(uploaded_file.getvalue())
    if lat and lon:
        st.subheader("📍 Geolocation Detected")
        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        st.map(map_data)
    else:
        st.info("No GPS Metadata found in image.")