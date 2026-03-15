from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    # Convert text to TF-IDF
    text_vector = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_vector)[0]

    # Probability
    probability = model.predict_proba(text_vector)[0]

    ai_score = probability[1] * 100
    human_score = probability[0] * 100

    if prediction == 1:
        result = "AI Generated"
    else:
        result = "Human Written"

    return {
        "result": result,
        "ai_confidence": round(ai_score, 2),
        "human_confidence": round(human_score, 2)
    }

if __name__ == "__main__":
    app.run(debug=True)