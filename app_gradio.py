import pickle
import numpy as np
import gradio as gr


def load_model():
    try:
        model = pickle.load(open("randomforest.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Error loading model or scaler: {e}")


model, scaler = load_model()


encoding_map = {
    "No": 0, "Yes": 1,
    "Negative": 0, "Positive": 1,
    "Low": 0, "Moderate": 1, "High": 2,
    "Unhealthy": 0, "Healthy": 1,
    "Non-Smoker": 0, "Smoker": 1
}


def predict_gradio(blood_glucose, insulin, bmi, age, bp, chol, waist,
                   glucose_test, pancreas, liver,
                   family, gene, activity, diet, smoking, alcohol, early):
    input_data = np.array([[
        blood_glucose, insulin, bmi, age, bp, chol, waist,
        encoding_map[family], encoding_map[gene], encoding_map[activity],
        encoding_map[diet], encoding_map[smoking], encoding_map[alcohol],
        glucose_test, pancreas, liver, encoding_map[early]
    ]])

    input_scaled = scaler.transform(input_data)
    probs = model.predict_proba(input_scaled)[0]
    pred = model.predict(input_scaled)[0]

    class_idx = np.where(model.classes_ == pred)[0][0]
    confidence = probs[class_idx] * 100

    top_idx = np.argsort(probs)[-3:][::-1]

    color = "black"
    if confidence > 70:
        color = "green"
    elif confidence > 50:
        color = "orange"

    html = f"<h2 style='color:{color}'>Prediction: {pred}</h2>"
    html += f"<p><strong>Confidence:</strong> {confidence:.1f}%</p>"
    html += "<h4>Top 3 possibilities</h4><ol>"
    for idx in top_idx:
        html += f"<li>{model.classes_[idx]} — {probs[idx]*100:.1f}%</li>"
    html += "</ol>"
    html += "<hr><p><em>Disclaimer:</em> This tool is for educational purposes only. Consult a medical professional for diagnosis.</p>"

    return html


def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🩺 Diabetes Type Prediction (Gradio)")
        with gr.Row():
            with gr.Column():
                blood_glucose = gr.Number(value=100, label="Blood Glucose (mg/dL)")
                insulin = gr.Number(value=20, label="Insulin Levels (μU/mL)")
                bmi = gr.Number(value=25.0, label="BMI")
                age = gr.Number(value=30, label="Age")
                bp = gr.Number(value=120, label="Blood Pressure (mmHg)")
                chol = gr.Number(value=200, label="Cholesterol (mg/dL)")
            with gr.Column():
                waist = gr.Number(value=80, label="Waist Circumference (cm)")
                glucose_test = gr.Number(value=100, label="Glucose Tolerance Test")
                pancreas = gr.Number(value=50, label="Pancreatic Health")
                liver = gr.Number(value=50, label="Liver Function")
                family = gr.Dropdown(choices=["No", "Yes"], value="No", label="Family History")
                gene = gr.Dropdown(choices=["Negative", "Positive"], value="Negative", label="Genetic Markers")
            with gr.Column():
                activity = gr.Dropdown(choices=["Low", "Moderate", "High"], value="Moderate", label="Physical Activity")
                diet = gr.Dropdown(choices=["Unhealthy", "Healthy"], value="Healthy", label="Diet Quality")
                smoking = gr.Dropdown(choices=["Non-Smoker", "Smoker"], value="Non-Smoker", label="Smoking")
                alcohol = gr.Dropdown(choices=["Low", "Moderate", "High"], value="Low", label="Alcohol")
                early = gr.Dropdown(choices=["No", "Yes"], value="No", label="Early Onset Symptoms")

        predict_btn = gr.Button("🔍 Predict")
        output = gr.HTML()

        predict_btn.click(fn=predict_gradio,
                          inputs=[blood_glucose, insulin, bmi, age, bp, chol, waist,
                                  glucose_test, pancreas, liver,
                                  family, gene, activity, diet, smoking, alcohol, early],
                          outputs=output)

    return demo


if __name__ == "__main__":
    iface = create_interface()
    iface.launch(server_name="0.0.0.0", share=False)
