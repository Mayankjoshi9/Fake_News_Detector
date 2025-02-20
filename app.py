from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)
load_dotenv()

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk_download('stopwords')
    stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
regex = re.compile('[^a-zA-Z]')

model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_trf")

try:
    model = joblib.load('model/fake_news_model.pkl')
    vectorizer = joblib.load('model/tfid_vectorizer.pkl')
except FileNotFoundError:
    print("Model or vectorizer file not found. Please check your file paths.")

def stemming(content):
    stemmed_content = regex.sub(' ', content).lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

def fetch_news(query):
    today = datetime.today()
    one_month = today - relativedelta(months=1)
    one_month_str = one_month.strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&from={one_month_str}&to={today_str}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def fetch_entity(text):
    doc = nlp(text)
    entity_set = set()
    essential_entities = ["PERSON", "FAC", "GPE", "LOC", "NORP", "EVENT", "LAW", "PRODUCT", "WORK_OF_ART"]
    for entity in doc.ents:
        if entity.label_ in essential_entities:
            entity_set.add(entity.text)
    return " ".join(entity_set) if entity_set else "world"

# ...existing code...

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']
    entity_text = fetch_entity(title)
    news_data = fetch_news(entity_text)
    high_articles = []
    highest_cosine = 0

    if 'articles' in news_data:
        for article in news_data['articles']:
            title1 = article.get("title", "")
            description1 = article.get("description", "")
            article1 = title1 + description1
            article2 = title + text
            embeddings = model1.encode([article1, article2])
            cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            highest_cosine = max(highest_cosine, cosine_sim)
            if cosine_sim > 0.5:
                high_articles.append(article)

    input_text = stemming(title + " " + text)
    input_text_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_text_tfidf)[0]

    if prediction == 0 and highest_cosine >= 0.3:
        predict = "Verified News"
    elif prediction == 0 and highest_cosine <= 0.1:
        predict = "Questionable"
    elif prediction == 1 and highest_cosine >= 0.5:
        predict = "Likely True"
    elif prediction == 1 and highest_cosine >= 0.2:
        predict = "Potentially Misleading"
    else:
        predict = "Fake News"
    
    return jsonify({"prediction": predict, "articles": high_articles})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)