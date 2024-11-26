import streamlit as st
import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import download as nltk_download
import requests
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk_download('stopwords')
    stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
regex = re.compile('[^a-zA-Z]')

@st.cache_resource
def load_transformer_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

model1 = load_transformer_model()
nlp = load_spacy_model()


try:
    model = joblib.load('model/fake_news_model.pkl')
    vectorizer = joblib.load('model/tfid_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check your file paths.")


st.title("Fake News Predictor")
st.subheader("Enter News Title and Text")


title = st.text_input("Title")
text = st.text_area("Content")

def stemming(content):
    """Preprocess content by cleaning, lowering, and stemming"""
    stemmed_content = regex.sub(' ', content).lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

def fetch_news(query):
    """Fetch news articles based on a query"""
    one_month = datetime.today() - relativedelta(months=1)
    date_str = one_month.strftime('%Y-%m-%d')
    api_key = os.getenv("NEWS_API_KEY")  # Replace with a safer environment variable in practice
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&from={date_str}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        with open('api_news/requests.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        st.write("Please wait for few moments...")
    except requests.exceptions.RequestException as e:
        st.error(f"Error in API request: {e}")

def fetch_entity(text):
   doc=nlp(text)
   entity_set=set()
   for entity in doc.ents:
        essential_entities = [
    "PERSON",
    "FAC",
    "GPE",
    "LOC",
    "NORP",
    "EVENT",
    "LAW",
    "PRODUCT",
    "WORK_OF_ART" ]

        if entity.label_ in essential_entities:
           entity_set.add(entity.text)
   entity_text=""
   for entity in entity_set:
       entity_text+=entity+" "
   
   return entity_text



if "feedback_label" not in st.session_state:
    st.session_state.feedback_label = None


if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False


if st.button("Predict"):
    
    entity_text=fetch_entity(title)
    if entity_text=="":
        entity_text="world"
    fetch_news(entity_text)
    highest_cosine=0
    with open("api_news/requests.json","r") as f:
        t=json.load(f)
        articles=t["articles"]
        for article in articles:
            article1=article["title"]+article["description"]
            article2=title+text
            embeddings = model1.encode([article1, article2])
            cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
            if(cosine_sim[0][0]>highest_cosine):
               highest_cosine=cosine_sim[0][0]



    input_text = stemming(title + " " + text)
    input_text_tfidf = vectorizer.transform([input_text])
    st.session_state.input_tfidf = input_text_tfidf

    try:
        prediction = model.predict(input_text_tfidf)
        predict=""
        if(prediction[0]==0 and highest_cosine>=0.3):
           predict="Verified News"    
        elif(prediction[0]==0 and highest_cosine<=0.1):
            predict="Questionable"
        elif(prediction[0]==1 and highest_cosine>=0.5):
            predict="Likely True"
        elif(prediction[0]==1 and highest_cosine>=0.2):
            predict="Potentially Misleading"
        else :
            predict="Fake News"
        if(predict=="Likely True" or predict=="Verified News"):
            st.markdown(f"### **✅ Prediction : {predict}**")
        elif (predict=="Questionable"):
            st.markdown(f"### **❓ Prediction : {predict}**")
        else:
            st.markdown(f"### **❌ Prediction : {predict}**")
            




        st.session_state.prediction_made = True  
    except Exception as e:
        st.error(f"Error in prediction: {e}")


if st.session_state.prediction_made:
    feedback_option = st.radio("Is this prediction correct? If not, select the correct label:",
                               ("No, it's Real News", "No, it's Fake News"))
    st.session_state.feedback_label = 0 if feedback_option == "No, it's Real News" else 1

    if st.session_state.feedback_label is not None:
        if st.button("Update Model with Feedback"):
            if hasattr(model, "partial_fit"):
                try:
                    
                    model.partial_fit(st.session_state.input_tfidf, [st.session_state.feedback_label], classes=[0, 1])
                    joblib.dump(model, 'model/fake_news_model.pkl')
                    joblib.dump(vectorizer, 'model/tfid_vectorizer.pkl')
                    st.write("The model has been updated with your feedback.")
                    st.session_state.prediction_made = False  
                    st.session_state.feedback_label = None 
                    st.session_state.input_tfidf=None
                except Exception as e:
                    st.error(f"Error updating model: {e}")
            else:
                st.warning("This model does not support online learning; it cannot be updated with new data.")