# main.py
import os
import requests
import numpy as np
import groq
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from io import BytesIO
from pypdf import PdfReader

# --- Initialize API Clients from Environment Variables ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"API key error: {e}") # Log error for debugging

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI()

# --- Helper Functions ---
def process_document(url: str):
    # ... (same chunking function as before)
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

def get_embeddings(chunks: list, task_type: str):
    response = genai.embed_content(model='models/embedding-001', content=chunks, task_type=task_type)
    return response['embedding']

def generate_answer(question: str, context: str):
    # ... (same Groq generation function as before)
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    chat_completion = groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.0)
    return chat_completion.choices[0].message.content.strip()

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    try:
        chunks = process_document(request.documents)
        chunk_embeddings = get_embeddings(chunks, "RETRIEVAL_DOCUMENT")
        
        all_answers = []
        for question in request.questions:
            question_embedding = get_embeddings([question], "RETRIEVAL_QUERY")[0]
            
            similarities = [np.dot(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
            context = "\n\n---\n\n".join([chunks[i] for i in top_indices])
            
            answer = generate_answer(question, context)
            all_answers.append(answer)
            
        return HackRxResponse(answers=all_answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
