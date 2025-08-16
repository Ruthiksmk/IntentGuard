
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load trained model and embedder
model = joblib.load("xgb_model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.title("🤖 Intention vs. Action Contradiction Detector")
st.write("Check if someone's action contradicts what they said they would do.")

# Input text boxes
intention = st.text_input("🧠 What was said (Intention)?", "I will avoid junk food today.")
action = st.text_input("💬 What was done (Action)?", "I had pizza and soda for lunch.")

if st.button("Predict"):
    # Get embeddings as 1D arrays
    intention_emb = embedder.encode(intention)    # shape: (embedding_dim,)
    action_emb = embedder.encode(action)          # shape: (embedding_dim,)

    # Reshape embeddings for cosine similarity calculation
    intention_emb_2d = intention_emb.reshape(1, -1)
    action_emb_2d = action_emb.reshape(1, -1)
    cos_sim = cosine_similarity(intention_emb_2d, action_emb_2d)[0][0]  # scalar

        # Concatenate features to match training (384 + 1 = 385)
    X = np.concatenate([
        intention_emb.ravel(),
        np.array([cos_sim])
    ])

    # Reshape for model input
    X = X.reshape(1, -1)


    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)
    confidence = np.max(proba) * 100

    result = "✅ Aligned" if prediction == 0 else "❌ Contradiction"
    st.subheader("Prediction:")
    st.success(f"{result} (Confidence: {confidence:.1f}%)")
