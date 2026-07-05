import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources (first time only)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load FAQ dataset
df = pd.read_csv("Data/faq.csv")


# -----------------------------
# Text Preprocessing Function
# -----------------------------
def preprocess(text):
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Preprocess all questions
df["Processed"] = df["Question"].apply(preprocess)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["Processed"])


# -----------------------------
# Get Answer Function
# -----------------------------
def get_answer(user_question):

    processed = preprocess(user_question)

    user_vector = vectorizer.transform([processed])

    similarity = cosine_similarity(user_vector, question_vectors)

    index = similarity.argmax()

    score = similarity[0][index]

    if score < 0.30:
        return "❌ Sorry, I couldn't find an appropriate answer."

    return df.iloc[index]["Answer"]