import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns



# Define file paths
file_paths = [
    "/content/Sentences_50Agree.txt",
    "/content/Sentences_66Agree.txt",
    "/content/Sentences_75Agree.txt",
    "/content/Sentences_AllAgree.txt"
]

# Load data from each file
data = []
for file_path in file_paths:
    with open(file_path, "r", encoding="ISO-8859-1") as file:  # Use ISO-8859-1 encoding
        lines = file.readlines()
        for line in lines:
            # Split each line into headline and sentiment
            headline, sentiment = line.strip().split("@")
            data.append({"headline": headline, "sentiment": sentiment})

# Convert to DataFrame
df = pd.DataFrame(data)

# Check the first few rows
#print(df.head())

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")  # Download the punkt_tab resource


# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning to the dataset
df["cleaned_headline"] = df["headline"].apply(clean_text)


# Encode sentiment labels
label_encoder = LabelEncoder()
df["sentiment_encoded"] = label_encoder.fit_transform(df["sentiment"])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_headline"])
y = df["sentiment_encoded"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probabilities for log loss

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Log Loss
loss = log_loss(y_test, y_pred_proba)
print(f"Log Loss: {loss:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Binarize the labels for multi-class ROC and Precision-Recall curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Adjust classes based on your label encoding
n_classes = y_test_bin.shape[1]

# ROC Curve for multi-class
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.show()

# Precision-Recall Curve for multi-class
from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    avg_precision = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(recall, precision, label=f"Class {label_encoder.classes_[i]} (AP = {avg_precision:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend()
plt.show()

# Evaluate on training set
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_loss = log_loss(y_train, y_train_pred_proba)

# Evaluate on test set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_loss = log_loss(y_test, y_test_pred_proba)

print("Training Accuracy:", train_accuracy)
print("Training Log Loss:", train_loss)
print("Test Accuracy:", test_accuracy)
print("Test Log Loss:", test_loss)

# Save Model
import joblib

# Save the model
model_filename = "financial_sentiment_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Save the LabelEncoder
label_encoder_filename = "label_encoder.pkl"
joblib.dump(label_encoder, label_encoder_filename)
print(f"Label encoder saved to {label_encoder_filename}")

# Save the TF-IDF Vectorizer
vectorizer_filename = "tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_filename)
print(f"TF-IDF Vectorizer saved to {vectorizer_filename}")
