
# main.py
import os
import requests
import numpy as np
import groq
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from pypdf import PdfReader  # <-- CORRECTED LINE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Initialize API Client from Environment Variables ---
try:
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    # This will help debug in the deployment logs if the key is missing
    print(f"CRITICAL: Failed to initialize Groq client. Check GROQ_API_KEY. Error: {e}")
    groq_client = None

# --- Pydantic Models for API data structure ---
class HackRxRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI()

# --- Helper Functions ---
def process_document(url: str):
    """Downloads and chunks the document."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        
        chunks = []
        chunk_size, chunk_overlap = 2000, 300
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - chunk_overlap
        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        print(f"Error processing document: {e}")
        return []

def generate_answer(question: str, context: str, groq_client_instance: groq.Groq):
    """Generates an answer using the provided Groq client instance."""
    if not groq_client_instance:
        raise HTTPException(status_code=500, detail="Groq client was not valid.")
        
    prompt = f"""
    You are an expert Q&A system. Your answers must be based *only* on the provided context.
    If the answer cannot be found in the context, state that clearly. Don't mention that you have read the document, it should be a direct one liner answer.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    try:
        chat_completion = groq_client_instance.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API call failed: {e}")
        if "authentication" in str(e).lower():
            raise HTTPException(status_code=401, detail="Authentication failed with Groq. Check your API key.")
        raise HTTPException(status_code=500, detail="Failed to generate answer from Groq.")

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(
    request: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    api_key = authorization.split(" ")[1]
    if not api_key:
        raise HTTPException(status_code=401, detail="Bearer token is empty.")

    try:
        # Initialize the Groq client for this specific request
        local_groq_client = groq.Groq(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Groq client: {e}")

    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' field in the request.")

    try:
        chunks = process_document(request.documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not process document from URL.")

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        document_matrix = vectorizer.fit_transform(chunks)

        all_answers = []
        for question in request.questions:
            question_vector = vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, document_matrix).flatten()
            
            top_k = 5
            top_indices = similarities.argsort()[-min(top_k, len(chunks)):][::-1]
            
            relevant_indices = [i for i in top_indices if similarities[i] > 0]
            if not relevant_indices:
                context = "No relevant context found."
            else:
                context = "\n\n---\n\n".join([chunks[i] for i in relevant_indices])
            
            answer = generate_answer(question, context, local_groq_client)
            all_answers.append(answer)
            
        return HackRxResponse(answers=all_answers)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An unexpected error occurred in the main endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    # The global groq_client doesn't exist anymore, so we remove that check.
    return {"status": "ok", "authentication_method": "Bearer Token"}
