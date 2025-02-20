---

## Fake News Detection Model with SDG Classifier & Streamlit App
[visit now](https://fakenewsbuster.streamlit.app/)

This repository contains a **Fake News Detection Model** built using an **SDG (Stochastic Gradient Descent) classifier** for classifying news articles as either fake or real. The model is trained on a publicly available dataset of news articles and is capable of distinguishing between fake and true news based on their content.

### Features:

- **Fake News Detection Model**: The core model uses an SDG classifier to predict whether a given news article is fake or true. The model is trained on a labeled dataset of news articles and uses natural language processing (NLP) techniques for feature extraction and prediction.
  
- **Streamlit App**: A user-friendly Streamlit web app that allows users to input news articles or reviews. The app then displays the model's prediction (fake or real) and prompts the user to provide feedback on whether the prediction was correct or not.

- **Feedback Loop**: Users can give feedback on the model's predictions (correct or incorrect). This feedback is then used to continuously improve the model by retraining it with the newly gathered data, creating a dynamic feedback loop for model enhancement.

### Installation:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fake-news-detection.git
    ```
    
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

### How It Works:

1. **Data Preprocessing**: The news dataset is cleaned and preprocessed, with text features like headlines and content being combined into a single column.
  
2. **Model Training**: The model is trained using the Stochastic Gradient Descent classifier to classify news articles based on their content.
  
3. **User Interaction**: The Streamlit app takes user input (news article or review) and provides a prediction on whether it’s fake or real.

4. **Model Improvement**: The app allows users to mark the model's prediction as correct or incorrect. Incorrect predictions are collected and used to retrain the model, improving its accuracy over time.

### Technologies Used:

- **Python**  
- **Streamlit** (for building the app)  
- **Scikit-learn** (for the SDG classifier)  
- **Pandas** and **NumPy** (for data handling)  
- **Natural Language Processing (NLP)** (for text feature extraction)

### Future Improvements:

- Enhance the feedback system by implementing a more sophisticated user interface.
- Use more advanced classifiers like **XGBoost** or **Deep Learning** models to improve prediction accuracy.
- Incorporate additional data sources to increase the model's robustness and coverage.

---

