import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open("randomforest.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()


# Header
st.title("🩺 Diabetes Type Prediction")
st.markdown("### Predict diabetes type using machine learning")
st.markdown("---")

# Input form
with st.form("prediction_form"):
    st.subheader("📋 Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Clinical Measurements**")
        blood_glucose = st.number_input("Blood Glucose (mg/dL)", 60, 300, 100)
        insulin = st.number_input("Insulin Levels (μU/mL)", 5, 80, 20)
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
        age = st.number_input("Age", 0, 100, 30)
        bp = st.number_input("Blood Pressure (mmHg)", 80, 180, 120)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 300, 200)
    
    with col2:
        st.markdown("**Tests & Assessments**")
        waist = st.number_input("Waist Circumference (cm)", 50, 150, 80)
        glucose_test = st.number_input("Glucose Tolerance Test", 50, 200, 100)
        pancreas = st.number_input("Pancreatic Health", 10, 90, 50)
        liver = st.number_input("Liver Function", 10, 90, 50)
    
    with col3:
        st.markdown("**Medical History & Lifestyle**")
        family = st.selectbox("Family History", ["No", "Yes"])
        gene = st.selectbox("Genetic Markers", ["Negative", "Positive"])
        activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        diet = st.selectbox("Diet Quality", ["Unhealthy", "Healthy"])
        smoking = st.selectbox("Smoking", ["Non-Smoker", "Smoker"])
        alcohol = st.selectbox("Alcohol", ["Low", "Moderate", "High"])
        early = st.selectbox("Early Onset Symptoms", ["No", "Yes"])
    
    st.markdown("---")
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

# Process prediction
if submitted:
    # Encode categorical features
    encoding_map = {
        "No": 0, "Yes": 1,
        "Negative": 0, "Positive": 1,
        "Low": 0, "Moderate": 1, "High": 2,
        "Unhealthy": 0, "Healthy": 1,
        "Non-Smoker": 0, "Smoker": 1
    }
    
    # Create input array (17 features)
    input_data = np.array([[
        blood_glucose, insulin, bmi, age, bp, chol, waist,
        encoding_map[family], encoding_map[gene], encoding_map[activity],
        encoding_map[diet], encoding_map[smoking], encoding_map[alcohol],
        glucose_test, pancreas, liver, encoding_map[early]
    ]])
    
    try:
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Show results
        st.markdown("---")
        st.subheader("✨ Prediction Results")
        
        # Main prediction
        predicted_type = prediction[0]
        confidence = probabilities[np.where(model.classes_ == predicted_type)[0][0]] * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if confidence > 70:
                st.success(f"### {predicted_type}")
                st.metric("Confidence", f"{confidence:.1f}%")
            elif confidence > 50:
                st.warning(f"### {predicted_type}")
                st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.info(f"### {predicted_type}")
                st.metric("Confidence", f"{confidence:.1f}%")
        
        with col2:
            # Top 3 predictions
            st.markdown("**Alternative Possibilities:**")
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            
            for i, idx in enumerate(top_3_idx):
                if i > 0:  # Skip first (already shown)
                    prob = probabilities[idx] * 100
                    st.write(f"{i+1}. {model.classes_[idx]} ({prob:.1f}%)")
        
        # Disclaimer
        st.markdown("---")
        st.info("⚠️ **Disclaimer:** This is a prediction tool for educational purposes. Always consult healthcare professionals for medical diagnosis.")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.write("This app uses a Random Forest model trained on diabetes patient data.")
    st.markdown("**Model Accuracy:** 85%")
    st.markdown(f"**Diabetes Types:** {len(model.classes_)}")
    
    with st.expander("📊 Detected Types"):
        for dt in sorted(model.classes_):
            st.write(f"• {dt}")
    
    st.markdown("---")
    st.markdown("**🔬 How it works:**")
    st.write("1. Enter patient data")
    st.write("2. AI analyzes patterns")
    st.write("3. Get diabetes type prediction")
    st.write("4. See confidence scores")
