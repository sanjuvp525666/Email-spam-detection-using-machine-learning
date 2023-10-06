import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Set page title and layout
st.set_page_config(page_title="Email Spam Classifier", layout="wide")

# Define CSS styles
main_bg_color = "linear-gradient(to bottom, #FFFFFF, #F1F3F4)"
title_font = {"font-size": "36px", "font-weight": "bold"}
subtitle_font = {"font-size": "18px", "color": "#707070", "margin-top": "10px"}
input_bg_color = "#E6E6E6"
predict_button_bg_color = "#4B8BBE"
predict_button_text_color = "#FFFFFF"
result_text_color = "#FFFFF"
footer_bg_color = "#0E1117" 
footer_text_color = "#FAFAFA"

# Define CSS classes
css = f"""
    <style>
    .stApp {{
        background-color: {main_bg_color};
    }}
    .title {{
        {title_font}
    }}
    .subtitle {{
        {subtitle_font}
    }}
    .input_textbox {{
        background-color: {input_bg_color};
        padding: 12px;
        height: 200px;
    }}
    .predict_button {{
        background-color: {predict_button_bg_color};
        color: {predict_button_text_color};
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 4px;
        cursor: pointer;
    }}
    .result_text {{
        color: {result_text_color};
        font-size: 24px;
        margin-top: 16px;
        padding: 8px;
        border-radius: 4px;
    }}
    .footer {{
        background-color: {footer_bg_color};
        color: {footer_text_color};
        padding: 16px;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
    }}
    </style>
"""

# Load CSS styles
st.markdown(css, unsafe_allow_html=True)

# Set title and subtitle
st.markdown("<h1 class='title'>Email Spam Classifier</h1>", unsafe_allow_html=True)

# Define function to preprocess text
def transform_text(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    filtered_words = []
    for word in message:
        if word.isalnum():
            filtered_words.append(word)

    message = filtered_words[:]
    filtered_words.clear()

    for word in message:
        if word not in stopwords.words('english') and word not in string.punctuation:
            filtered_words.append(word)

    message = filtered_words[:]
    filtered_words.clear()

    for word in message:
        filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)


# Load model and vectorizer
tfidf_vectorizer = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Get user input
input_message = st.text_area("Paste your message below:", key="input_email", value="", height=200)

# Perform classification on button click
if st.button("CHECK", key="predict_button"):
    # Preprocess text
    transformed_text = transform_text(input_message)

    # Vectorize text
    vectorized_text = tfidf_vectorizer.transform([transformed_text])

    # Make prediction
    prediction = model.predict(vectorized_text)[0]

    # Display prediction result
    if prediction == 'spam':
        st.markdown("<p class='result_text' style='background-color: red;'>SPAM</p>", unsafe_allow_html=True)
    elif prediction == "ham":
        st.markdown("<p class='result_text' style='background-color: green;'>NOT SPAM</p>", unsafe_allow_html=True)
    else:
        st.error("Error occurred during prediction.")

# Add footer
st.markdown("<div class='footer'>Built with ❤️</div>", unsafe_allow_html=True)
