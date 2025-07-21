# 📊 Churn Prediction Dashboard

A machine learning-powered web application built with Streamlit to predict customer churn based on user attributes and behavior. This project provides insights into churn risk, visualizes important metrics, and helps organizations make data-driven retention decisions.

---

## 🚀 Features

- 📈 Interactive Dashboard built using **Streamlit**
- 🧠 Integrated with trained **Machine Learning model** (`scikit-learn`)
- 📊 Rich visualizations using **Matplotlib**, **Seaborn**, and **Plotly**
- 📝 Upload your own CSV datasets for live prediction
- 📌 View prediction results, performance metrics, and charts
---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**:  
  - `pandas`, `numpy` – for data manipulation  
  - `matplotlib`, `seaborn`, `plotly` – for data visualization  
  - `scikit-learn` – for training & prediction    
  - `joblib` – for saving/loading the ML model

---

## 📂 Project Structure

churn_prediction/
├── Dashboard.py # Streamlit app entry point
├── model.pkl # Trained ML model file
├── data/
│ └── sample_data.csv # Example dataset
├── src/
│ ├── preprocessing.py # Preprocessing functions
│ └── utils.py # Utility/helper functions
├── requirements.txt # Required Python packages
└── README.md # Project overview
---

## ⚙️ Getting Started

### 1. 🔧 Installation

```bash
git clone https://github.com/yourusername/churn_prediction.git
cd churn_prediction
pip install -r requirements.txt
2. ▶️ Run the App
streamlit run Dashboard.py
📈 Output Visuals
✅ Confusion Matrix

✅ ROC-AUC Curve

✅ Feature Importance

✅ SHAP Summary Plot

✅ Prediction Summary Table

💡 How it Works
Upload Dataset – Upload a CSV file with customer data.

Preprocessing – Data is cleaned and encoded using preprocessing.py.

Prediction – Model makes churn predictions with probability.

Visualization – Results are visualized with multiple charts.

Interpretation – SHAP values explain each prediction.

📘 Dependencies
Listed in requirements.txt:
streamlit
pandas
numpy
matplotlib
seaborn
plotly
shap
scikit-learn
joblib

Install them using:
pip install -r requirements.txt
🙌 Acknowledgments
Scikit-learn
Streamlit
Matplotlib
eSaborn
Plotly

👨‍💻 Author
Sahil Hingu
B.Tech IT | Shah & Anchor Kuttchi Engineering College
📧 Email: sahilhingu31@gmail.com
