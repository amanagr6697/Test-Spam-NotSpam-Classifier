import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
#importing required libraries

#Creating instance to run stemming on words
ps = PorterStemmer()


#Text preprocessing and returns text after removing punctuations, special characaters and after performing stemming
def transform_text(text):
    #Making text more understandable
    text = text.lower()
    #Breaking text into list of words
    text = nltk.word_tokenize(text)
    
    y = []
    #if words are not special characters append them
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    #Removing punvtuations and stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    #Stemming
    for i in text:
        y.append(ps.stem(i))
    
    #Returning as string
    return " ".join(y)

#Loading vectorizer for words and model to predict
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
#Heading for site
st.title("Text Spam Classifier")

input_sms = st.text_area("Enter message that you have to check for Spam/Not Spam")

if st.button('Predict Spam/Not Spam'):

    # 1. Preprocess the data 
    transformed_sms = transform_text(input_sms)
    # 2. vectorize it so that it can work on our model
    vector_input = tfidf.transform([transformed_sms])
    # 3. Give result according to the predictions
    result = model.predict(vector_input)[0]
    # 4. Display the result got onto the screen
    if result == 1:
        st.header("This message is a Spam message.")
    else:
        st.header("This is not a Spam message.")