# ============================================================
# ZENTHOR - MAIN.PY
# Horizontal General Purpose AI
# Groq Text + Groq Vision + Groq Whisper + RAG (Safe)
# ============================================================

import base64
import io
import os
import re
import tempfile
from datetime import datetime  # ✅ এই লাইনটি অবশ্যই থাকতে হবে
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile
)

from fastapi.responses import HTMLResponse

from groq import Groq

import uvicorn

# ai_brain থেকে ask_ai এবং rag ইমপোর্ট করুন
from ai_brain import ask_ai, rag

from config import (
    GROQ_API_KEY
)


# ============================================================
# APP
# ============================================================

app = FastAPI(

    title="Zenthor",
    description="Zenthor - General Purpose Horizontal AI Assistant",
    version="5.0.0"

)


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = None

if GROQ_API_KEY:

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# MODELS
# ============================================================

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"
WHISPER_MODEL = "whisper-large-v3"


# ============================================================
# CHAT MEMORY
# ============================================================

chat_sessions = {}


# ============================================================
# CLEAN AI RESPONSE
# ============================================================

def clean_ai_response(text: str):

    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=(re.DOTALL | re.IGNORECASE)
    )

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(contents: bytes):

    try:

        import pypdf

        reader = pypdf.PdfReader(
            io.BytesIO(contents)
        )

        pages = []

        for page in reader.pages:

            try:

                text = page.extract_text()
                if text:
                    pages.append(text)

            except Exception:
                continue

        return "\n\n".join(pages).strip()

    except Exception as e:

        return f"Unable to extract PDF text: {str(e)}"


# ============================================================
# SESSION
# ============================================================

def get_session_history(session_id: str):

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    return chat_sessions[session_id]


def save_message(session_id: str, role: str, content: str):

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({
        "role": role,
        "content": content
    })

    # Keep latest 40 messages.
    chat_sessions[session_id] = chat_sessions[session_id][-40:]


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)

async def home():

    try:

        index_file = (

            Path(__file__)
            .resolve()
            .parent
            /
            "index.html"

        )

        if not index_file.exists():

            return HTMLResponse(

                f"""
                <html>
                <body style="
                    background:#131314;
                    color:white;
                    font-family:Arial;
                    padding:40px;
                ">

                    <h2>Zenthor frontend not found.</h2>
                    <p>Expected file:</p>
                    <code>{index_file}</code>

                </body>
                </html>
                """,

                status_code=500

            )

        html = index_file.read_text(
            encoding="utf-8"
        )

        return HTMLResponse(

            content=html,
            status_code=200

        )

    except Exception as e:

        return HTMLResponse(

            f"""
            <html>
            <body style="
                background:#131314;
                color:white;
                font-family:Arial;
                padding:40px;
            ">

                <h2>Zenthor Frontend Error</h2>
                <p>{str(e)}</p>

            </body>
            </html>
            """,

            status_code=500

        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {

        "status": "ok",
        "service": "Zenthor",
        "version": "6.0.0",
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "voice_model": WHISPER_MODEL,
        "gemini": False,
        "serper": bool(
            os.environ.get("SERPER_API_KEY", "")
        ),
        "web_search": bool(
            os.environ.get("SERPER_API_KEY", "")
        ),
        "mode": "horizontal-ai-with-web-search-and-rag"

    }


# ============================================================
# VOICE TRANSCRIPTION
# ============================================================

@app.post("/api/transcribe")

async def transcribe_audio(audio: UploadFile = File(...)):

    temp_audio_path = None

    try:

        if (
            not GROQ_API_KEY
            or not groq_client
        ):

            return {
                "status": "error",
                "message": "GROQ_API_KEY missing."
            }

        audio_content = await audio.read()

        if not audio_content:
            return {
                "status": "error",
                "message": "Empty audio file."
            }

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm"
        ) as temp_audio:

            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name

        with open(temp_audio_path, "rb") as audio_file:

            transcription = (

                groq_client
                .audio
                .transcriptions
                .create(

                    file=(
                        os.path.basename(temp_audio_path),
                        audio_file.read()
                    ),

                    model=WHISPER_MODEL,
                    response_format="json"

                )

            )

        return {
            "status": "success",
            "text": transcription.text or ""
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

    finally:

        if (
            temp_audio_path
            and
            os.path.exists(temp_audio_path)
        ):

            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


# ============================================================
# SMART CHAT TITLE
# ============================================================

@app.post("/api/generate-title")

async def generate_title(prompt: str = Form(...)):

    try:

        prompt = (prompt or "").strip()

        if not prompt:
            return {
                "status": "success",
                "title": "New Chat"
            }

        if (
            not GROQ_API_KEY
            or not groq_client
        ):

            return {
                "status": "success",
                "title": create_fallback_title(prompt)
            }

        title_prompt = f"""
You are naming a chat conversation.

Create a short, natural title based ONLY on the
user's first message.

USER MESSAGE:
{prompt}

RULES:

- 2 to 5 words only.
- Maximum about 40 characters.
- Capture the main intent/topic.
- Do not answer the user's question.
- Do not write a sentence.
- Do not use quotation marks.
- Do not use emojis.
- Do not add punctuation at the end.
- Do not write "Chat", "New Chat", "Question",
  "User Question" or similar generic words.
- Match the user's language when possible.
- For Bangla/Banglish users, a natural Bangla or
  Banglish title is preferred.
- Output ONLY the title.
"""

        completion = (

            groq_client
            .chat
            .completions
            .create(

                model=TEXT_MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": title_prompt
                    }
                ],

                max_completion_tokens=30,
                temperature=0.2

            )

        )

        title = (
            completion
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        title = clean_title(title)

        if not title:
            title = create_fallback_title(prompt)

        return {
            "status": "success",
            "title": title
        }

    except Exception:

        return {
            "status": "success",
            "title": create_fallback_title(prompt)
        }


# ============================================================
# CLEAN TITLE
# ============================================================

def clean_title(title):

    if not title:
        return ""

    title = str(title).strip()

    title = re.sub(
        r"^[\"'`]+|[\"'`]+$",
        "",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    title = re.sub(
        r"[.!?。！？]+$",
        "",
        title
    )

    title = re.sub(

        r"^(title|chat title|conversation title)\s*:\s*",
        "",
        title,
        flags=re.IGNORECASE

    )

    words = title.split()

    if len(words) > 5:
        title = " ".join(words[:5])

    return title[:60].strip()


# ============================================================
# FALLBACK TITLE
# ============================================================

def create_fallback_title(prompt):

    prompt = (prompt or "").strip()

    if not prompt:
        return "New Chat"

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"[?؟]+$", "", prompt)

    words = prompt.split()

    if not words:
        return "New Chat"

    title_words = words[:5]
    title = " ".join(title_words)

    return title[:50].strip()


# ============================================================
# GROQ VISION
# ============================================================

async def analyze_image(

    question: str,
    image_bytes: bytes,
    mime_type: str

):

    if (
        not GROQ_API_KEY
        or not groq_client
    ):

        return (None, "GROQ_API_KEY missing.")

    if (
        len(image_bytes)
        >
        20 * 1024 * 1024
    ):

        return (
            None,
            "Image is too large. Please upload an image smaller than 20 MB."
        )

    try:

        image_base64 = (
            base64
            .b64encode(image_bytes)
            .decode("utf-8")
        )

        image_url = (
            f"data:{mime_type};base64,{image_base64}"
        )

        user_question = (question or "").strip()

        if not user_question:
            user_question = "What is shown in this image?"

        prompt = f"""
You are Zenthor, a general-purpose horizontal AI assistant.

Look at the uploaded image and answer the user's question
directly.

USER QUESTION:
{user_question}

RESPONSE STYLE:

- Answer the exact question first.
- Do not automatically describe the entire image.
- Do not automatically create sections such as Subject,
  Analysis, Answer, Description or Conclusion.
- Use natural sentences and short paragraphs.
- Keep simple questions simple.
- If the user asks for detailed analysis, provide details.
- If a list or steps are useful, use them naturally.
- Match the user's language.
- If the user writes Bangla/Banglish, answer naturally in Bangla.
- Do not repeat the user's question.
- Do not invent information.
- Only state what can reasonably be determined from the image.
- Do not infer sensitive personal attributes from appearance.
"""

        response = (

            groq_client
            .chat
            .completions
            .create(

                model=VISION_MODEL,

                messages=[

                    {

                        "role": "user",

                        "content": [

                            {
                                "type": "text",
                                "text": prompt
                            },

                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }

                        ]

                    }

                ],

                temperature=0.3,
                max_completion_tokens=500,
                stream=False

            )

        )

        answer = (
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        if not answer:
            return (None, "Groq Vision returned an empty response.")

        return (clean_ai_response(answer), None)

    except Exception as e:

        return (None, f"Groq Vision error: {str(e)}")


# ============================================================
# MAIN CHAT API
# ============================================================

@app.post("/api/chat")

async def chat_endpoint(

    question: str = Form(""),
    session_id: str = Form("default"),
    grade: str = Form(""),
    subject: str = Form(""),
    file: UploadFile = File(None)

):

    try:

        question = (question or "").strip()
        history = get_session_history(session_id)

        # ====================================================
        # FILE UPLOAD
        # ====================================================

        if file and file.filename:

            filename = file.filename.lower()
            file_contents = await file.read()

            # =================================================
            # PDF
            # =================================================

            if (

                filename.endswith(".pdf")
                or
                file.content_type == "application/pdf"

            ):

                extracted_text = extract_pdf_text(file_contents)

                if not extracted_text:
                    return {
                        "status": "error",
                        "message": "No readable text found in PDF."
                    }

                # =============================================
                # SAVE TO RAG (✅ সম্পূর্ণ সুরক্ষিত)
                # =============================================
                if rag is not None:
                    try:
                        rag.add_document(
                            extracted_text,
                            metadata={
                                "filename": file.filename,
                                "user": session_id,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        print(f"RAG: Document added successfully - {file.filename}")
                    except Exception as e:
                        print(f"RAG Error: {str(e)}")
                else:
                    print("RAG is not available. Document not stored in vector DB.")

                # =============================================
                # CONTINUE WITH REGULAR CHAT
                # =============================================
                user_question = (

                    question

                    or

                    "Please summarize this document "
                    "and explain the most important "
                    "information."

                )

                document_context = (

                    f"User uploaded a PDF named: {file.filename}\n\n"
                    f"Document content:\n{extracted_text[:12000]}"

                )

                if grade:
                    document_context += f"\n\nUser level/context: {grade}"
                if subject:
                    document_context += f"\nSubject/context: {subject}"

                response = ask_ai(

                    user_question=user_question,
                    chat_history=history,
                    context_text=document_context

                )

                response = clean_ai_response(response)

                save_message(
                    session_id,
                    "user",
                    f"[PDF: {file.filename}] {user_question}"
                )

                save_message(
                    session_id,
                    "assistant",
                    response
                )

                return {
                    "status": "success",
                    "question": user_question,
                    "response": response
                }

            # =================================================
            # IMAGE
            # =================================================

            if (

                file.content_type
                and
                file.content_type.startswith("image/")

            ):

                user_question = (

                    question

                    or

                    "What is shown in this image?"

                )

                image_response, error = await analyze_image(

                    user_question,
                    file_contents,
                    file.content_type

                )

                if error:
                    return {
                        "status": "error",
                        "message": error
                    }

                save_message(
                    session_id,
                    "user",
                    f"[Image: {file.filename}] {user_question}"
                )

                save_message(
                    session_id,
                    "assistant",
                    image_response
                )

                return {
                    "status": "success",
                    "question": user_question,
                    "response": image_response
                }

            # =================================================
            # UNSUPPORTED FILE
            # =================================================

            return {
                "status": "error",
                "message": "Unsupported file type."
            }

        # ====================================================
        # NORMAL TEXT CHAT
        # ====================================================

        if not question:
            return {
                "status": "error",
                "message": "Please enter a message."
            }

        contextual_question = question

        if grade:
            contextual_question += f"\n\n[Optional user context: {grade}]"
        if subject:
            contextual_question += f"\n[Optional subject context: {subject}]"

        response = ask_ai(
            user_question=contextual_question,
            chat_history=history
        )

        response = clean_ai_response(response)

        save_message(session_id, "user", question)
        save_message(session_id, "assistant", response)

        return {
            "status": "success",
            "question": question,
            "response": response
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
