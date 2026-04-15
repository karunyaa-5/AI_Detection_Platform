import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score


# -----------------------
# Load Dataset
# -----------------------
data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]


# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------
# TF-IDF Vectorization
# -----------------------
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------
# Logistic Regression
# -----------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))


# -----------------------
# SVM
# -----------------------
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_vec, y_train)

svm_pred = svm_model.predict(X_test_vec)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))


# -----------------------
# Random Forest
# -----------------------
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)

rf_pred = rf_model.predict(X_test_vec)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))


# -----------------------
# Ensemble (Soft Voting)
# -----------------------
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('svm', svm_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

ensemble_model.fit(X_train_vec, y_train)

ensemble_pred = ensemble_model.predict(X_test_vec)
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))


# -----------------------
# Save Final Model
# -----------------------
pickle.dump(ensemble_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nFinal Ensemble Model Saved Successfully!")