import pandas as pd
import os
import urllib.request
import json

class FAQChatbot:
    def __init__(self):
       
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(BASE_DIR, "Data", "faq.csv")
        self.df = None

        
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
               
                self.df.columns = self.df.columns.str.strip().str.lower()
            except Exception as e:
                print(f"Backend Warning: Could not parse CSV: {e}")
        else:
            print(f"Backend Warning: CSV database not found at {self.csv_path}")

     
        self.api_key = os.getenv("GEMINI_API_KEY", "YOUR_FREE_GEMINI_API_KEY_HERE")

    def _call_gemini_fallback(self, query):
        """
        Backend Fallback Engine: If the CSV doesn't have the answer,
        this method calls Google Gemini to generate an answer.
        Uses raw urllib so you don't have to install extra dependencies.
        """
        if not self.api_key or self.api_key == "YOUR_FREE_GEMINI_API_KEY_HERE":
            return "I couldn't find an answer in the FAQ database, and the AI API key is not configured."

        url = f"https://googleapis.com{self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        
        prompt_payload = {
            "contents": [{
                "parts": [{"text": f"You are a helpful company chatbot assistant. Answer this user question briefly and professionally: {query}"}]
            }]
        }

        try:
            data = json.dumps(prompt_payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
                return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"I couldn't find that in our local database, and the live AI backup server is currently unreachable."

    def reply(self, user_query):
        """
        Main Engine Logic:
        1. Validates inputs
        2. Searches local CSV data (Fuzzy and string containment matching)
        3. Falls back to live AI if local data doesn't contain the answer.
        """
       
        if not user_query or not user_query.strip():
            return "Please type a valid question."

        clean_query = user_query.strip().lower()

        
        if self.df is not None and 'question' in self.df.columns and 'answer' in self.df.columns:
            try:
               
                matches = self.df[self.df['question'].str.lower().str.contains(clean_query, na=False, regex=False)]
                if not matches.empty:
                    return matches['answer'].iloc[0]

                
                for _, row in self.df.iterrows():
                    stored_q = str(row['question']).lower()
                    if stored_q in clean_query:
                        return row['answer']
                        
            except Exception as e:
                print(f"Backend execution warning during local lookup: {e}")

        # Step B: Advanced AI Processing
        # If we reach this line, the local CSV file didn't have the answer
        return self._call_gemini_fallback(user_query)
