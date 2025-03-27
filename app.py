import streamlit as st
import joblib
import pandas as pd
import re
import nltk

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")  # Download the punkt_tab resource


# Load the model, vectorizer, and label encoder
model = joblib.load("financial_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize & remove stopwords
    return " ".join(tokens)

# Streamlit app configuration
st.set_page_config(page_title="Financial News Sentiment Analyzer", page_icon="📈")

# Title and description
st.title("📊 Financial News Sentiment Analyzer")
st.write("""
Welcome to the Financial News Sentiment Analyzer! 🎉  
This app predicts the sentiment of financial news headlines as **positive**, **negative**, or **neutral**.  
Use the text box below to enter a headline, or upload a CSV file for batch predictions.
""")

# Sidebar information
st.sidebar.header("About")
st.sidebar.write("""
This app uses a **machine learning model** trained on financial news data to predict sentiment.  
The model is based on **Logistic Regression** and uses **TF-IDF** for text vectorization.
""")

st.sidebar.header("Model Performance Metrics 📊")
st.sidebar.write("**Training Accuracy:** 0.9312 🎯")
st.sidebar.write("**Training Log Loss:** 0.2906 📉")
st.sidebar.write("**Test Accuracy:** 0.8992 🎯")
st.sidebar.write("**Test Log Loss:** 0.3409 📉")

# Input options
option = st.radio("Choose an option:", ("Enter a headline", "Upload a CSV file"))

if option == "Enter a headline":
    headline = st.text_input("Enter a financial news headline:")

    if headline:
        cleaned_headline = clean_text(headline)  # Clean text
        vectorized_headline = vectorizer.transform([cleaned_headline])  # Transform with TF-IDF
        prediction = model.predict(vectorized_headline)  # Predict sentiment
        predicted_sentiment = label_encoder.inverse_transform(prediction)[0]

        # Display the result with emojis
        st.write("---")
        st.subheader("Prediction Result 🎯")
        sentiment_dict = {
            "positive": ("success", "😊"),
            "negative": ("error", "😠"),
            "neutral": ("info", "😐")
        }
        st_func, emoji = sentiment_dict.get(predicted_sentiment, ("info", "🤔"))
        getattr(st, st_func)(f"Predicted Sentiment: **{predicted_sentiment}** {emoji}")

else:
    uploaded_file = st.file_uploader("Upload a CSV file with a 'headline' column:", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "headline" not in df.columns:
            st.error("The CSV file must contain a column named 'headline'.")
        else:
            df["cleaned_headline"] = df["headline"].apply(clean_text)  # Clean all headlines
            vectorized_headlines = vectorizer.transform(df["cleaned_headline"])  # Transform all at once
            df["predicted_sentiment"] = label_encoder.inverse_transform(model.predict(vectorized_headlines))  # Predict

            # Display results
            st.write("---")
            st.subheader("Batch Prediction Results 📋")
            st.write(df[["headline", "predicted_sentiment"]])

            # Download results as CSV
            st.download_button(
                label="Download Predictions as CSV",
                data=df[["headline", "predicted_sentiment"]].to_csv(index=False),
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

# Footer
st.write("---")
st.write("Made with ❤️ by [Taqi Javed]")
