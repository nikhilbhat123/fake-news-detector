import streamlit as st
import pickle
import string

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# UI
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article to check if it's Real or Fake")

# Input box
user_input = st.text_area("Enter News Text:")

if st.button("Check"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")
    else:
        st.warning("Please enter some text")