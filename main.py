import os

import pymupdf
import requests
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ai-twins.vercel.app",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
async def root():
    return {"message": "Backend is running"}


def extract_text(file_name, content):
    if file_name.endswith(".txt"):
        return content.decode("utf-8")

    if file_name.endswith(".pdf"):
        pdf = pymupdf.open(stream=content, filetype="pdf")
        text = ""

        for page in pdf:
            text += page.get_text()

        return text

    if file_name.endswith(".docx"):
        from io import BytesIO

        document = Document(BytesIO(content))

        text = ""

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

        return text

    return ""


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    input: str = Form(...),
    file: UploadFile = File(None)
):
    document_text = ""

    if file:
        content = await file.read()
        document_text = extract_text(file.filename, content)

    prompt = f"""Use the following document to answer the user's question.
                Document:{document_text}
                Question:{input}"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "inclusionai/ling-3.0-flash-fin:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
    )

    result = response.json()

    return {"response": result["choices"][0]["message"]["content"]}
