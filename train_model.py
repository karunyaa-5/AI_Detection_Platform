import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from xgboost import XGBClassifier

print("Step 1: Loading dataset...")

# Load dataset
data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]

print("Dataset loaded successfully")
print("Total samples:", len(data))


print("Step 2: Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


print("Step 3: Vectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorization completed")


print("Step 4: Initializing models...")

lr = LogisticRegression()
svm = SVC(probability=True, kernel="linear")
rf = RandomForestClassifier(n_estimators=50)

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss"
)

print("Models initialized")


print("Step 5: Creating ensemble model...")

ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("svm", svm),
        ("rf", rf),
        ("xgb", xgb)
    ],
    voting="soft"
)

print("Ensemble model ready")


print("Step 6: Training model... (this may take a few minutes)")

ensemble.fit(X_train_vec, y_train)

print("Training completed")


print("Step 7: Evaluating model...")

accuracy = ensemble.score(X_test_vec, y_test)
print("Model Accuracy:", accuracy)


print("Step 8: Saving model...")

pickle.dump(ensemble, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")
print("Training pipeline finished.")