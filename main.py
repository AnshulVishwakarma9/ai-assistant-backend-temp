import json
import os

import pymupdf
import requests
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ai-twins.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Backend is running"}


def extract_text(file_name, content):
    file_name = file_name.lower()

    if file_name.endswith(".txt"):
        return content.decode("utf-8")

    if file_name.endswith(".pdf"):
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)

    if file_name.endswith(".docx"):
        from io import BytesIO

        document = Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    return ""


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    input: str = Form(...),
    file: UploadFile = File(None),
    generateImage: str = Form("false"),
):
    try:
        payload = json.loads(input)
    except json.JSONDecodeError:
        payload = {"currentChatQuestion": input}

    question = payload.get("currentChatQuestion", input)
    previous_messages = payload.get("previousChatHistory", [])
    user_info = payload.get("userInfo", {})

    prompt = f"""User information: {json.dumps(user_info)}
Question: {question}"""

    if file:
        content = await file.read()
        document_text = extract_text(file.filename or "", content)

        if not document_text:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use TXT, PDF, or DOCX.",
            )

        prompt = f"""Use the following document to answer the user's question.

Document:
{document_text}

Question: {question}"""

    if generateImage.lower() == "true":
        prompt = f"""Generate valid SVG code for the user's description.
Return only the SVG code without Markdown fences.
Description: {question}"""

    messages = [
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in previous_messages[-5:]
        if item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
    ]

    messages.append({
        "role": "user",
        "content": prompt,
    })

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openrouter/free",
                "messages": messages,
            },
            timeout=60,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Unable to connect to OpenRouter.",
        )

    if response.status_code != 200:
        print("OpenRouter status:", response.status_code)
        print("OpenRouter response:", response.text)

        raise HTTPException(
            status_code=502,
            detail="OpenRouter request failed.",
        )

    result = response.json()

    try:
        answer = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Invalid response received from OpenRouter.",
        )

    return {"response": answer}
