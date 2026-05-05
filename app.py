from flask import Flask, render_template, request, jsonify, session
import pytesseract
import platform
from PIL import Image
import os
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import re  
import google.generativeai as genai 

app = Flask(__name__)
app.secret_key = 'prognosxai_clinical_secret_key_2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- AI Configuration ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- Tesseract OCR Configuration ---
# This check prevents the app from crashing on Render's Linux servers
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

FEATURE_NAMES = [
    'Glucose', 'HbA1c', 'Insulin', 'BMI', 'Cholesterol', 'Triglycerides', 
    'RBC', 'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'RDW', 'WBC', 
    'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils', 
    'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Troponin', 'CRP'
]

# --- Load Clinical Models ---
try:
    with open(os.path.join(BASE_DIR, 'xgb_model1.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler1.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print(" SUCCESS: Clinical models and scalers loaded.")
except Exception as e:
    print(f" ERROR: Initialization failed: {e}")

# --- FIDELITY LOGIC ---
def calculate_fidelity_score(ai_text, shap_info):
    if shap_info == 'Not provided' or not ai_text:
        return 0
    required_markers = [m.strip().lower() for m in shap_info.split(',')]
    found_count = 0
    for marker in required_markers:
        if marker in ai_text.lower():
            found_count += 1
    return (found_count / len(required_markers)) * 100

# --- WEB NAVIGATION ROUTES ---

@app.route('/')
def home(): return render_template('index.html')

@app.route('/login')
def login(): return render_template('login.html')

@app.route('/register/doctor')
def reg_doc(): return render_template('register_doctor.html')

@app.route('/register/lab')
def reg_lab(): return render_template('register_lab.html')

@app.route('/doctor/dashboard')
def doctor_dashboard(): return render_template('doctor_dashboard.html')

@app.route('/lab/dashboard')
def lab_dashboard(): return render_template('lab_dashboard.html')

@app.route('/patient/database')
def patient_db(): return render_template('patient_database.html')

@app.route('/settings')
def settings_page(): return render_template('settings.html')

@app.route('/logout')
def logout_confirm():
    session.clear()
    return render_template('logout.html')

@app.route('/admin/panel')
def admin_panel(): return render_template('admin_panel.html')

@app.route('/patient/profile/<patient_id>')
def patient_profile(patient_id):
    return render_template('profile.html')

# --- CORE CLINICAL API ROUTES ---

@app.route('/get-explanation', methods=['POST'])
def get_explanation():
    try:
        data = request.get_json()
        disease = data.get('disease')
        language = data.get('language', 'English')
        shap_info = data.get('shap_info', 'Not provided')
        
        # IMPROVED PROGRESSION LOGIC:
        # Front-end should send 'previous_disease' by looking at the patient's history in the table
        prev_disease = data.get('previous_disease', 'None (Initial Visit)')
        
        progression_prompt_chunk = ""
        if prev_disease and prev_disease != 'None (Initial Visit)' and prev_disease != disease:
            progression_prompt_chunk = f"""
            DISEASE PROGRESSION ANALYSIS:
            The patient was previously diagnosed with {prev_disease}. 
            Explain how {prev_disease} potentially contributed to or evolved into the current state of {disease}.
            Focus on how the biomarkers {shap_info} illustrate this transition.
            """
        else:
            progression_prompt_chunk = "This is a baseline diagnosis. Explain the condition based on the current markers."

        prompt = f"""
        Role: Clinical AI Interpretability Specialist for PrognosXAI.
        Patient Status: Predicted Condition is {disease}.
        Mathematical Evidence (SHAP): {shap_info}.
        
        {progression_prompt_chunk}

        CRITICAL REQUIREMENT: 
        1. Respond strictly in the {language} language.
        2. MATHEMATICAL FIDELITY: You MUST prioritize the biomarkers in the EXACT order provided in: {shap_info}.
        3. Use professional clinical terminology.

        Provide a professional clinical explanation using HTML:
        - <h4>Condition Overview</h4>: Define {disease}.
        - <h4>Progression Analysis</h4>: Explain the clinical shift from {prev_disease} to {disease}. (If {prev_disease} is None, focus on baseline onset).
        - <h4>SHAP Graph Interpretation</h4>: Explain how the AI prioritized {shap_info} mathematically.
        - <h4>Future Health Risks</h4>: Risks based on {shap_info}.
        - <h4>Targeted Control Strategy</h4>: Actionable protocols.
        - <h4>Emergency Warning Signs</h4>: Red flags.

        Tone: Scientific and Authoritative.
        """
        
        response = gemini_model.generate_content(prompt)
        explanation_text = response.text if response.text else ""

        accuracy = calculate_fidelity_score(explanation_text, shap_info)
        
        return jsonify({
            "explanation": explanation_text if explanation_text else "<p>Error generating insights.</p>",
            "prompt_accuracy": accuracy
        })
    except Exception as e:
        print(f" Gemini API Error: {e}")
        return jsonify({"explanation": "<p>Clinical interpretation temporarily unavailable.</p>"}), 500

@app.route('/ocr-extract', methods=['POST'])
def extract_ocr():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        img = Image.open(file.stream)
        text = pytesseract.image_to_string(img)
        
        extracted_data = {}
        for marker in FEATURE_NAMES:
            pattern = rf"{marker}[:\s\-]+(\d+\.?\d*)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted_data[marker] = match.group(1)
                
        return jsonify(extracted_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        if not features or len(features) != 24:
            return jsonify({"diagnosis": "Error", "error": "24 clinical inputs required"}), 400

        float_features = [float(x) for x in features]
        input_df = pd.DataFrame([float_features], columns=FEATURE_NAMES)
        
        scaled_features = scaler.transform(input_df)
        prediction_idx = int(model.predict(scaled_features)[0])
        
        disease_map = {0: "Anemia", 1: "Diabetes", 2: "Healthy", 3: "Infection", 4: "Thalasseamia", 5: "Heart Disease"}
        diagnosis = disease_map.get(prediction_idx, "Unknown Condition")

        img_base64 = ""
        shap_summary = ""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled_features)

            if isinstance(shap_values, list):
                current_shap = shap_values[prediction_idx]
            elif len(shap_values.shape) == 3:
                current_shap = shap_values[0, :, prediction_idx]
            else:
                current_shap = shap_values[0]

            top_indices = np.argsort(np.abs(current_shap))[-5:][::-1]
            shap_summary = ", ".join([f"{FEATURE_NAMES[i]}" for i in top_indices])

            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap.Explanation(values=current_shap, data=scaled_features[0], feature_names=FEATURE_NAMES), show=False)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        except Exception as shap_err:
            print(f" SHAP Calculation Error: {shap_err}")

        return jsonify({
            "diagnosis": diagnosis,
            "shap_image": img_base64,
            "shap_info": shap_summary  
        })
    except Exception as e:
        return jsonify({"diagnosis": "Model Error", "error": str(e)}), 500

if __name__ == '__main__':
    # Set debug=False to save memory and prevent OpenBLAS errors
    app.run(debug=False, port=5000)