import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pymupdf

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
                  "http://localhost:5173", 
                  "http://127.0.0.1:5173",
                  "https://ai-chat-app-av-dev.vercel.app"
                    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


def extract_text(file_name, content):
    if file_name.endswith(".txt"):
        return content.decode("utf-8")

    if file_name.endswith(".pdf"):
        pdf = pymupdf.open(stream=content, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    return ""


@app.post("/api/chat")
async def chat(input: str = Form(...), file: UploadFile = File(None)):

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
