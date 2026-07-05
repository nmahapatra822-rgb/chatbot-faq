import streamlit as st
import pandas as pd
import os
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


st.markdown("""
<style>
.stApp {
    background-color: #050505;
    background-image: 
        radial-gradient(circle at 10% 0%, rgba(218, 5, 115, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(118, 15, 215, 0.12) 0%, transparent 45%);
    background-attachment: fixed;
}
.main { background-color: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background-color: #0b0b0d !important; }
[data-testid="stSidebar"] div[data-testid="stImage"] { background: transparent !important; padding: 0 !important; }

.robot-container { text-align: center; margin: 20px 0; }
.cute-robot {
    font-size: 75px;
    display: inline-block;
    animation: jumpRobot 1.2s ease-in-out infinite alternate;
}
@keyframes jumpRobot {
    0% { transform: translateY(0) scale(1); }
    100% { transform: translateY(-25px) scale(0.95); }
}

.title {
    text-align: left;
    color: #ffffff !important;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-top: 10px;
    line-height: 1.2;
}
.welcome-title { text-align: center; color: #ffffff !important; font-size: 36px; font-weight: 800; margin-bottom: 5px; }
.subtitle { text-align: left; color: #6e6e73 !important; font-size: 15px; margin-bottom: 30px; }
.welcome-subtitle {
    text-align: center; color: #da0573 !important; font-weight: bold; font-size: 18px; margin-bottom: 25px;
}

.user {
    background: linear-gradient(135deg, #da0573 0%, #760fd7 100%) !important;
    color: #ffffff !important;
    padding: 14px 18px;
    border-radius: 20px;
    border-bottom-right-radius: 4px;
    margin: 12px 0;
    font-size: 15px;
}
.bot {
    background: #121214 !important;
    color: #e2e2e6 !important;
    padding: 14px 18px;
    border-radius: 20px;
    border-bottom-left-radius: 4px;
    margin: 12px 0;
    border: 1px solid #1c1c1f;
    font-size: 15px;
}

div[data-testid="stTextInput"] input {
    background-color: #121214 !important;
    color: #ffffff !important;
    border: 1px solid #1c1c1f !important;
    border-radius: 25px !important;
    padding: 12px 20px !important;
}
div[data-testid="stTextInput"] input:focus { border-color: #da0573 !important; }

div.stButton > button {
    background: linear-gradient(135deg, #da0573 0%, #760fd7 100%) !important;
    color: #ffffff !important;
    border-radius: 20px !important;
    border: none !important;
    padding: 10px 32px !important;
    font-weight: 600 !important;
}
.footer { text-align: center; color: #444446; margin-top: 60px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "history" not in st.session_state: st.session_state.history = []


def preprocess_text(text):
    """Fulfills Task Requirement: Tokenizes, cleans, and normalizes text."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = text.split()            
    return " ".join(tokens)

def get_nlp_backend_reply(user_query, csv_path):
    """Processes queries using Text Preprocessing and Cosine Similarity."""
    if not user_query or not user_query.strip():
        return "Please type a valid question."

    if not os.path.exists(csv_path):
        return "Database System Error: The FAQ knowledge base data file is missing."

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()

        if 'question' not in df.columns or 'answer' not in df.columns:
            return "Database format error: Columns must include 'question' and 'answer'."

    
        cleaned_user_query = preprocess_text(user_query)

        
        cleaned_questions = df['question'].apply(preprocess_text).tolist()

        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_questions + [cleaned_user_query])

        
        cosine_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        
        best_match_idx = cosine_scores.argmax()
        highest_similarity = cosine_scores[best_match_idx]

        
        if highest_similarity > 0.30:
            return df['answer'].iloc[best_match_idx]
        
    except Exception as e:
        return f"An internal analytical pipeline error occurred: {str(e)}"

    return "I couldn't find an answer to that specific question in my FAQ database. Could you please rephrase it or try another topic?"



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "Data", "faq.csv")


if not st.session_state.logged_in:
    st.markdown('<div class="robot-container"><span class="cute-robot">🤖</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-title">Welcome to My World!</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Let\'s get your profile set up</p>', unsafe_allow_html=True)
    
    name_input = st.text_input("Enter your Name")
    
    st.markdown('<div style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
    if st.button("Enter Chatbot World"):
        if name_input.strip() != "":
            st.session_state.user_name = name_input.strip()
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Please fill in your Name to enter.")
    st.markdown('</div>', unsafe_allow_html=True)


else:
    with st.sidebar:
        st.image("assets/chatbot_logo.png", width=180)
        st.title("FAQ Chatbot")
        st.write(f"Logged in as: **{st.session_state.user_name}**")
        
        if st.button("🚪 Logout / Switch User"):
            st.session_state.logged_in = False
            st.session_state.history = []
            st.rerun()
            
        if st.button("🗑 Clear Chat"):
            st.session_state.history = []

    st.markdown(f'<p class="title">Hey {st.session_state.user_name},<br>explore & be inspired</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask anything from the FAQ database.</p>', unsafe_allow_html=True)

    question = st.text_input("Type your question here...")

    if st.button("Ask"):
        if question.strip() != "":
            
            answer = get_nlp_backend_reply(question, CSV_FILE_PATH)
            st.session_state.history.append(("You", question))
            st.session_state.history.append(("Bot", answer))

    for sender, message in st.session_state.history:
        if sender == "You":
            st.markdown(f'<div class="user"><b>🧑 {st.session_state.user_name}:</b><br>{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot"><b>🤖 Bot:</b><br>{message}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
