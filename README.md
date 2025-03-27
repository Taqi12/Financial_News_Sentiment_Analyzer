# Financial_News_Sentiment_Analyzer 

This project builds a machine learning model to classify text sentiment into **Negative, Neutral, and Positive** categories. It uses a **Logistic Regression** classifier trained on a labeled dataset.  

## Usage  

1. Enter a text review in the input field.  
2. Click the **"Predict"** button to analyze the sentiment.  
3. The app will display whether the sentiment is **Negative, Neutral, or Positive**.  

## Dataset  

* The model is trained on a labeled text dataset.  
* The dataset should contain two columns: **"text"** (input) and **"sentiment"** (labels).  

## Model  

* **Logistic Regression** is used for classification.  
* **TF-IDF vectorization** is applied for feature extraction.  

## Dependencies  

* pandas  
* scikit-learn  
* nltk  
* streamlit  
* joblib  

## Notes  

* Adjust the dataset path and column names in `train_model.py` if necessary.  
* The model's performance depends on the dataset quality and preprocessing.  
* You can improve accuracy by trying different vectorization techniques or classifiers.  

## Author  

* [Taqi Javed]
