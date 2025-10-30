# 🩺 Diabetes Type Prediction System

A machine learning-powered web application that predicts diabetes types using Random Forest classification. The system can identify 13 different diabetes types with 85% accuracy.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)

## 🎯 Overview

This project uses machine learning to predict various types of diabetes based on clinical measurements, genetic markers, and lifestyle factors. The model can distinguish between 13 different diabetes types including Type 1, Type 2, Gestational Diabetes, LADA, MODY, and more.

## ✨ Features

- **Multi-class Classification**: Predicts 13 different diabetes types
- **Interactive Web App**: User-friendly Streamlit interface
- **High Accuracy**: 85% prediction accuracy on test data
- **Confidence Scores**: Shows probability for each prediction
- **Real-time Predictions**: Instant results based on patient data
- **Comprehensive Input**: 17 clinical and lifestyle features
- **Alternative Predictions**: Shows top 3 most likely diabetes types

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 84.98% |
| **Training Accuracy** | 100.00% |
| **Cross-Validation Score** | 84.7% (±0.7%) |
| **Number of Classes** | 13 |
| **Model Type** | Random Forest Classifier |

### Detected Diabetes Types:
1. Type 1 Diabetes
2. Type 2 Diabetes
3. Gestational Diabetes
4. LADA (Latent Autoimmune Diabetes in Adults)
5. MODY (Maturity Onset Diabetes of the Young)
6. Neonatal Diabetes Mellitus (NDM)
7. Prediabetic
8. Cystic Fibrosis-Related Diabetes (CFRD)
9. Secondary Diabetes
10. Steroid-Induced Diabetes
11. Type 3c Diabetes (Pancreatogenic Diabetes)
12. Wolfram Syndrome
13. Wolcott-Rallison Syndrome

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd diabitesss/myenv
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

Required packages:
```
streamlit
pandas
numpy
scikit-learn
pickle-mixin
```

4. **Verify model files exist**
```bash
ls *.pkl
# Should show: randomforest.pkl, scaler.pkl
```

## 💻 Usage

### Running the Web Application

1. **Start the Streamlit app**
```bash
streamlit run app.py
```

2. **Access the application**
- Open your browser and go to: `http://localhost:8501`
- The app will automatically open in your default browser

3. **Make a prediction**
   - Enter patient clinical measurements
   - Provide medical history and lifestyle information
   - Click "Predict" button
   - View the predicted diabetes type with confidence score

### Using the Jupyter Notebook

To train the model or explore the data:

```bash
jupyter notebook randomforest.ipynb
```

## 📁 Dataset

**File**: `data/diabetes_dataset00.csv`

**Features** (17 total):
- **Clinical Measurements**: Blood Glucose, Insulin Levels, BMI, Age, Blood Pressure, Cholesterol, Waist Circumference
- **Medical Tests**: Glucose Tolerance Test, Pancreatic Health, Liver Function Tests
- **Genetic/Medical History**: Family History, Genetic Markers, Early Onset Symptoms
- **Lifestyle Factors**: Physical Activity, Dietary Habits, Smoking Status, Alcohol Consumption

**Target**: Diabetes Type (13 classes)

## 🛠️ Technology Stack

- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Model Persistence**: pickle
- **Development**: Jupyter Notebook
- **Language**: Python 3.13

## 📂 Project Structure

```
diabitesss/myenv/
├── app.py                      # Streamlit web application
├── randomforest.ipynb          # Jupyter notebook for training
├── randomforest.pkl            # Trained model (serialized)
├── scaler.pkl                  # Feature scaler (serialized)
├── README.md                   # Project documentation
├── data/
│   └── diabetes_dataset00.csv  # Training dataset
├── bin/                        # Virtual environment binaries
├── lib/                        # Python libraries
└── requirements.txt            # Python dependencies
```

## 🧠 Model Details

### Algorithm: Random Forest Classifier

**Hyperparameters**:
- `n_estimators`: 100 (number of trees)
- `random_state`: 42 (for reproducibility)
- Default scikit-learn settings for other parameters

### Training Process:

1. **Data Preprocessing**
   - Label encoding for categorical features
   - StandardScaler for numerical features normalization

2. **Train-Test Split**
   - 80% training data
   - 20% test data
   - Random state: 42

3. **Model Training**
   - Random Forest with 100 decision trees
   - Multi-class classification
   - Cross-validation for robustness

### Feature Importance:
The model uses all 17 features with varying importance:
- Blood Glucose Levels
- Age
- BMI
- Genetic Markers
- Family History
- (and 12 more features)

## ⚠️ Known Issues & Limitations

1. **Overfitting**: Model shows 100% training accuracy vs 85% test accuracy (15% gap)
2. **Class Imbalance**: Some diabetes types may have fewer samples
3. **Feature Encoding**: Manual encoding required for categorical features
4. **Medical Disclaimer**: This is a prediction tool for educational purposes only

## 🔮 Future Improvements

- [ ] Implement feature selection to reduce overfitting
- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Include feature importance visualization
- [ ] Add more detailed medical recommendations
- [ ] Implement model versioning
- [ ] Add data validation and error handling
- [ ] Create API endpoints for integration
- [ ] Add patient history tracking
- [ ] Implement explainable AI (SHAP values)

## 📝 How to Retrain the Model

1. Open `randomforest.ipynb` in Jupyter
2. Run all cells sequentially
3. Model will be saved automatically to `randomforest.pkl` and `scaler.pkl`
4. Restart the Streamlit app to use the new model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚖️ Disclaimer

**This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.**

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated**: October 30, 2025  
**Model Version**: 1.0  
**Accuracy**: 84.98%
