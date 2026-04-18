import pickle
import string

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

while True:
    news = input("\nEnter news (or type exit): ")

    if news.lower() == "exit":
        break

    news = clean_text(news)   # ✅ ADD THIS
    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)

    if result[0] == 1:
        print("REAL NEWS")
    else:
        print("FAKE NEWS")