import streamlit as st
import joblib
import pandas as pd

# Load the model, vectorizer, and label encoder
model = joblib.load("financial_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text cleaning function (same as during training)
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Clean the text
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit app
st.set_page_config(page_title="Financial News Sentiment Analyzer", page_icon="📈")

# Title and description
st.title("📊 Financial News Sentiment Analyzer")
st.write("""
Welcome to the Financial News Sentiment Analyzer! 🎉
This app predicts the sentiment of financial news headlines as **positive**, **negative**, or **neutral**.
Use the text box below to enter a headline, or upload a CSV file for batch predictions.
""")

# Sidebar for additional info and metrics
st.sidebar.header("About")
st.sidebar.write("""
This app uses a **machine learning model** trained on financial news data to predict sentiment.
The model is based on **Logistic Regression** and uses **TF-IDF** for text vectorization.
You can upload a CSV file with a column named `headline` to analyze multiple headlines at once.
""")

# Display training and test metrics in the sidebar
st.sidebar.header("Model Performance Metrics 📊")
st.sidebar.write("**Training Accuracy:** 0.9312 🎯")
st.sidebar.write("**Training Log Loss:** 0.2906 📉")
st.sidebar.write("**Test Accuracy:** 0.8992 🎯")
st.sidebar.write("**Test Log Loss:** 0.3409 📉")

# Input options
option = st.radio("Choose an option:", ("Enter a headline", "Upload a CSV file"))

if option == "Enter a headline":
    # Input text box for a single headline
    headline = st.text_input("Enter a financial news headline:")

    if headline:
        # Clean the headline
        cleaned_headline = clean_text(headline)
        # Vectorize the headline
        vectorized_headline = vectorizer.transform([cleaned_headline])
        # Predict sentiment
        prediction = model.predict(vectorized_headline)
        predicted_sentiment = label_encoder.inverse_transform(prediction)[0]

        # Display the result with emojis
        st.write("---")
        st.subheader("Prediction Result 🎯")
        if predicted_sentiment == "positive":
            st.success(f"Predicted Sentiment: **{predicted_sentiment}** 😊")
        elif predicted_sentiment == "negative":
            st.error(f"Predicted Sentiment: **{predicted_sentiment}** 😠")
        else:
            st.info(f"Predicted Sentiment: **{predicted_sentiment}** 😐")

else:
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload a CSV file with a 'headline' column:", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        if "headline" not in df.columns:
            st.error("The CSV file must contain a column named 'headline'.")
        else:
            # Clean and predict sentiment for each headline
            df["cleaned_headline"] = df["headline"].apply(clean_text)
            df["vectorized_headline"] = df["cleaned_headline"].apply(lambda x: vectorizer.transform([x]))
            df["predicted_sentiment"] = df["vectorized_headline"].apply(lambda x: label_encoder.inverse_transform(model.predict(x))[0])

            # Display the results
            st.write("---")
            st.subheader("Batch Prediction Results 📋")
            st.write(df[["headline", "predicted_sentiment"]])

            # Download the results as a CSV file
            st.download_button(
                label="Download Predictions as CSV",
                data=df[["headline", "predicted_sentiment"]].to_csv(index=False),
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

# Footer
st.write("---")
st.write("Made with ❤️ by [Taqi Javed]")