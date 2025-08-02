# main.py

import os
import requests
import numpy as np
import groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional  # <-- Import Optional
from io import BytesIO
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Initialize API Client ---
try:
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"CRITICAL: Failed to initialize Groq client. Check GROQ_API_KEY. Error: {e}")
    groq_client = None

# --- Pydantic Models ---
# Make the request fields optional to handle edge cases from the judge
class HackRxRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI()

# --- Helper Functions (No changes here) ---
def process_document(url: str):
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

def generate_answer(question: str, context: str):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized.")
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        chat_completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API call failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer from Groq.")

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    # Add validation for the optional fields
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
            
            answer = generate_answer(question, context)
            all_answers.append(answer)
            
        return HackRxResponse(answers=all_answers)
    except Exception as e:
        print(f"An unexpected error occurred in the main endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "groq_client_initialized": groq_client is not None}
