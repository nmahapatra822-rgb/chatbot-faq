import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


df = pd.read_csv("Data/faq.csv")



def preprocess(text):
    text = text.lower()

   
    text = text.translate(str.maketrans('', '', string.punctuation))

    
    words = word_tokenize(text)

    
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)



df["Processed"] = df["Question"].apply(preprocess)


vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["Processed"])



def get_answer(user_question):

    processed = preprocess(user_question)

    user_vector = vectorizer.transform([processed])

    similarity = cosine_similarity(user_vector, question_vectors)

    index = similarity.argmax()

    score = similarity[0][index]

    if score < 0.30:
        return "❌ Sorry, I couldn't find an appropriate answer."

    return df.iloc[index]["Answer"]
