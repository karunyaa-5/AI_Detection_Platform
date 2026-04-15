from flask import Flask, render_template, request, jsonify
import pickle
import requests
import re
import os
app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔹 Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"
import os

API_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def check_with_api(text):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        return response.json()
    except:
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")

    if not text:
        return jsonify({"error": "No text provided"})

    # 🔹 LOCAL MODEL
    text_vector = vectorizer.transform([text])
    proba = model.predict_proba(text_vector)

    ai_prob_local = proba[0][1]
    human_prob_local = proba[0][0]

    # 🔹 API MODEL
    api_result = check_with_api(text)

    ai_prob_api = 0.5
    if api_result and isinstance(api_result, list):
        try:
            ai_prob_api = api_result[0][0]['score']
        except:
            pass

    # 🔥 HYBRID WEIGHT
    final_ai_prob = (0.4 * ai_prob_local) + (0.6 * ai_prob_api)

    # 🔥 DECISION LOGIC
    if final_ai_prob >= 0.65:
        result = "AI Generated"
    elif final_ai_prob <= 0.35:
        result = "Human Written"
    else:
        if ai_prob_api > 0.6:
            result = "AI Generated"
        elif ai_prob_api < 0.4:
            result = "Human Written"
        else:
            result = "Likely AI Generated"

    # 🔥 DETECTION TREND (Sentence-wise)
    sentences = re.split(r'[.!?]+', text)
    trend = []

    for sentence in sentences:
        if sentence.strip():
            vec = vectorizer.transform([sentence])
            prob = model.predict_proba(vec)[0][1]
            trend.append(round(prob * 100, 2))

    return jsonify({
        "result": result,
        "final_ai": round(final_ai_prob * 100, 2),
        "final_human": round((1 - final_ai_prob) * 100, 2),
        "local_ai": round(ai_prob_local * 100, 2),
        "api_ai": round(ai_prob_api * 100, 2),
        "trend": trend
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)