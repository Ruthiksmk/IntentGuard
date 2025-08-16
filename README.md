IntentGuard – Intention vs Action Contradiction Classifier
🔧 Tools & Technologies

Python · SentenceTransformers · Scikit-learn · XGBoost · Streamlit · Ngrok

📖 Overview

IntentGuard is an NLP-based system that classifies whether a person’s stated intention aligns or contradicts with their actual action.

Developed a custom Think–Do Contradiction dataset.

Leveraged SentenceTransformers for semantic embeddings.

Trained multiple ML models (Logistic Regression, Random Forest, XGBoost, MLP).

Achieved strong accuracy in alignment vs contradiction detection.

Deployed an interactive Streamlit web app for real-time predictions.

Tools & Technologies

Python

SentenceTransformers

Scikit-learn

XGBoost

Streamlit

Google Colab (for training & experiments)

Project Structure
IntentGuard/
│── app.py                           # Streamlit web app  
│── Think_Do_Contradiction_Dataset_Improved.csv   # Custom dataset  
│── xgb_model.pkl                     # Trained XGBoost model  
│── train_and_eval.ipynb              # Notebook with training + evaluation  
│── requirements.txt                  # Dependencies  
│── README.md                         # Project documentation  


git clone https://github.com/yourusername/IntentGuard.git
cd IntentGuard
pip install -r requirements.txt


streamlit run app.py


GOOGLE COLAB : https://colab.research.google.com/drive/1ltRyRerZrQv6Osumtm1QeNCzet8DUqLg?authuser=2#scrollTo=m_pQMkZf0PyE

Results

Sentence embeddings captured intent–action meaning effectively.

XGBoost achieved the best accuracy among tested models.

Interactive predictions possible via Streamlit UI.

Future Improvements

Expand dataset size for better generalization.

Deploy on Streamlit Cloud / Hugging Face Spaces for a public demo.

Add explainability features (SHAP, LIME).


## 📂 Dataset
The custom **Think–Do Contradiction Dataset** is stored in this repo:  
[Think_Do_Contradiction_Dataset_Improved.csv](Think_Do_Contradiction_Dataset_Improved.csv)

When running in **Google Colab**, load it directly from GitHub:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/yourusername/IntentGuard/main/Think_Do_Contradiction_Dataset_Improved.csv"
df = pd.read_csv(url)
