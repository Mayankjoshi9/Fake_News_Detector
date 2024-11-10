import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Load the model and vectorizer from disk
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfid_vectorizer.pkl')

# Streamlit UI
st.title("Fake News Predictor")
st.subheader("Enter News Title and Text")

# Inputs
title = st.text_input("Title")
text = st.text_area("Content")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  
regex = re.compile('[^a-zA-Z]')               

def stemming(content):
    stemmed_content = regex.sub(' ', content).lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)


if st.button("Predict"):
    # Preprocess and predict
    input_text =stemming( title + " " + text)
    input_text_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_text_tfidf)
    # y_proba = model.predict_proba(input_text_tfidf)[:,1]
    # threshold = 0.54
    st.write("Prediction:", "Real News" if (prediction[0]==0) else "Fake News")

    correct = st.radio("Is this prediction correct?", ("Yes", "No"))
    
    if st.button("submit"):

        if correct=="No":
            feedback_label=st.radio("What is the correct label?", ("Real News", "Fake News"))
            feedback_label= 0 if feedback_label=="Real News" else 1

            updated_model=model.partial_fit(input_text_tfidf,[feedback_label])

            joblib.dump(updated_model,'model/fake_news_model.pkl')
            joblib.dump(vectorizer, 'model/tfid_vectorizer.pkl')
            
        st.write("Thank you for your feedback! The model has been updated.")
        