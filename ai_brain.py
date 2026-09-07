# ============================================================
# ZENTHOR - HORIZONTAL AI BRAIN
# Groq + Serper Google Search + RAG (Safe Mode)
# ============================================================

import json
import re
import urllib.request
import urllib.error
from typing import Optional

from groq import Groq

from config import (
    GROQ_API_KEY,
    SERPER_API_KEY,
    SERPER_ENDPOINT,
    SERPER_RESULTS,
    SERPER_COUNTRY,
    SERPER_LANGUAGE,
    SYSTEM_PROMPT,
    PRIMARY_GROQ_MODEL,
    FALLBACK_GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    MAX_HISTORY_MESSAGES,
)

# ============================================================
# RAG ENGINE (Safe Import with Fallback)
# ============================================================

rag = None
RAG_AVAILABLE = False

try:
    from rag_engine import RAGEngine
    # Use in-memory mode to avoid disk permission issues on Render
    rag = RAGEngine(persist_directory=None)  # None = in-memory
    RAG_AVAILABLE = True
    print("RAG Engine loaded successfully (in-memory mode)")
except Exception as e:
    print(f"RAG Engine could not be loaded: {str(e)}")
    print("Zenthor will run without RAG functionality.")


# ============================================================
# GROQ CLIENT
# ============================================================

client = None

if GROQ_API_KEY:
    client = Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# MODEL CACHE
# ============================================================

_available_models = None


# ============================================================
# WEB SEARCH DETECTION
# ============================================================

CURRENT_KEYWORDS = [

    # English
    "latest",
    "today",
    "current",
    "now",
    "recent",
    "newest",
    "this week",
    "this month",
    "this year",

    "news",
    "update",
    "updates",

    "price",
    "cost",
    "worth",
    "net worth",
    "salary",
    "market cap",
    "stock price",

    "score",
    "scores",
    "standings",
    "ranking",
    "rankings",

    "release date",
    "released",
    "launch",
    "launched",

    "weather",
    "forecast",

    "president",
    "prime minister",

    # Bangla
    "আজ",
    "এখন",
    "বর্তমান",
    "সর্বশেষ",
    "সাম্প্রতিক",
    "নতুন খবর",
    "খবর",
    "আপডেট",
    "দাম",
    "মূল্য",
    "বেতন",
    "সম্পদ",
    "নেট ওয়ার্থ",
    "স্কোর",
    "র‍্যাংক",
    "র‌্যাঙ্ক",
    "আবহাওয়া",
    "কত টাকা",
    "বর্তমানে",
    "এই মুহূর্তে",
    "সর্বশেষ খবর",
]


# ============================================================
# SEARCH INTENT
# ============================================================

def needs_web_search(question: str) -> bool:

    if not question:
        return False

    text = str(question).strip().lower()

    explicit_patterns = [

        r"\bsearch\b",
        r"\bgoogle\b",
        r"\bweb\b",
        r"\bonline\b",
        r"\binternet\b",
        r"\blook up\b",
        r"\bfind out\b",

        r"গুগলে",
        r"সার্চ",
        r"ওয়েবে",
        r"ইন্টারনেটে",

    ]

    for pattern in explicit_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    for keyword in CURRENT_KEYWORDS:
        if keyword in text:
            return True

    dynamic_patterns = [

        r"\bhow much\b.*\b(now|today|currently)\b",
        r"\bwho is\b.*\b(current|now)\b",
        r"\bwhat is\b.*\b(current|now|today)\b",
        r"\bwhat's\b.*\b(current|now|today)\b",
        r"\bhow many\b.*\b(currently|today)\b",

        r"এখন.*কত",
        r"বর্তমানে.*কত",
        r"এখন.*কে",

    ]

    for pattern in dynamic_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


# ============================================================
# SERPER GOOGLE SEARCH
# ============================================================

def google_search(query: str) -> Optional[dict]:

    if not SERPER_API_KEY:
        return None

    query = (query or "").strip()

    if not query:
        return None

    payload = {
        "q": query,
        "gl": SERPER_COUNTRY,
        "hl": SERPER_LANGUAGE,
        "num": SERPER_RESULTS,
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        SERPER_ENDPOINT,
        data=data,
        method="POST",
        headers={
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
    )

    try:

        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            result = json.loads(raw.decode("utf-8"))
            return result

    except Exception as e:

        print("SERPER SEARCH ERROR:", str(e))
        return None


# ============================================================
# FORMAT SEARCH RESULTS
# ============================================================

def format_search_results(search_data: dict) -> str:

    if not search_data:
        return ""

    parts = []

    # Knowledge Graph
    knowledge = search_data.get("knowledgeGraph") or {}

    if knowledge:

        kg_lines = []

        title = knowledge.get("title")
        description = knowledge.get("description")
        type_name = knowledge.get("type")

        if title:
            kg_lines.append(f"Name: {title}")
        if type_name:
            kg_lines.append(f"Type: {type_name}")
        if description:
            kg_lines.append(f"Description: {description}")

        for key, value in knowledge.items():

            if key in ["title", "description", "type", "imageUrl", "website"]:
                continue

            if isinstance(value, str):
                kg_lines.append(f"{key}: {value}")

        if kg_lines:
            parts.append("KNOWLEDGE GRAPH:\n" + "\n".join(kg_lines))

    # Organic Results
    organic = search_data.get("organic") or []

    result_lines = []

    for index, item in enumerate(organic[:SERPER_RESULTS], start=1):

        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        link = item.get("link") or ""
        date = item.get("date") or ""

        line = f"[{index}] {title}"

        if date:
            line += f" ({date})"
        if snippet:
            line += f"\nSnippet: {snippet}"
        if link:
            line += f"\nURL: {link}"

        result_lines.append(line)

    if result_lines:
        parts.append("GOOGLE SEARCH RESULTS:\n" + "\n\n".join(result_lines))

    # Answer Box
    answer_box = search_data.get("answerBox") or {}

    if answer_box:

        answer_parts = []

        for key in ["title", "answer", "snippet"]:

            value = answer_box.get(key)
            if value:
                answer_parts.append(f"{key}: {value}")

        if answer_parts:
            parts.append("GOOGLE ANSWER BOX:\n" + "\n".join(answer_parts))

    return "\n\n".join(parts).strip()


# ============================================================
# GET AVAILABLE MODELS
# ============================================================

def get_available_models():

    global _available_models

    if _available_models is not None:
        return _available_models

    if not client:
        return []

    try:

        model_list = client.models.list()
        models = []

        for model in model_list.data:

            model_id = getattr(model, "id", None)
            if model_id:
                models.append(model_id)

        _available_models = models
        return models

    except Exception as e:

        print("MODEL LIST ERROR:", str(e))
        return []


# ============================================================
# CHOOSE MODELS
# ============================================================

def choose_models():

    available = get_available_models()

    preferred = [

        PRIMARY_GROQ_MODEL,
        FALLBACK_GROQ_MODEL,
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",

    ]

    result = []

    if available:

        for model_name in preferred:

            if model_name and model_name in available and model_name not in result:
                result.append(model_name)

        for model_name in available:

            if model_name not in result:
                result.append(model_name)

    else:

        for model_name in preferred:

            if model_name and model_name not in result:
                result.append(model_name)

    return result


# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_response(text):

    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=(re.DOTALL | re.IGNORECASE)
    )

    text = re.sub(
        r"^\s*(assistant|zenthor)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# ============================================================
# BUILD SYSTEM PROMPT
# ============================================================

def build_system_prompt(web_context: str = ""):

    prompt = str(SYSTEM_PROMPT or "").strip()

    if web_context:

        prompt += """

============================================================
LIVE WEB CONTEXT
============================================================

The following information was retrieved from Google Search.

Use this information when answering the user's question.

IMPORTANT:

- Prefer recent and relevant results.
- Do not invent unsupported details.
- If the results conflict, explain the uncertainty.

---------------- WEB RESULTS ----------------

""" + web_context + """

---------------- END WEB RESULTS ----------------
"""

    return prompt


# ============================================================
# ASK GROQ
# ============================================================

def _ask_groq(messages):

    if not client:
        return (None, "GROQ_API_KEY is not set.")

    models_to_try = choose_models()

    if not models_to_try:
        return (None, "No available Groq model found.")

    errors = []

    for model_name in models_to_try:

        try:

            response = (
                client
                .chat
                .completions
                .create(

                    model=model_name,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS,

                )

            )

            if response and response.choices:

                answer = response.choices[0].message.content

                if answer:
                    return (answer, None)

        except Exception as e:

            errors.append(f"{model_name}: {str(e)}")
            continue

    return (
        None,
        "Could not get response from Groq models.\n"
        + "\n".join(errors[-3:])
    )


# ============================================================
# MAIN AI ENGINE (with Safe RAG)
# ============================================================

def ask_ai(
    user_question,
    chat_history=None,
    context_text=None,
    use_rag=True,
):

    try:

        if not GROQ_API_KEY:
            return "GROQ_API_KEY is not set."

        final_question = (str(user_question).strip() if user_question else "")

        if not final_question:
            return "Please enter a message."

        # ====================================================
        # 1. WEB SEARCH
        # ====================================================

        web_context = ""

        if needs_web_search(final_question):

            print("WEB SEARCH:", final_question)

            search_data = google_search(final_question)

            if search_data:

                web_context = format_search_results(search_data)
                print("WEB SEARCH SUCCESS")

            else:

                print("WEB SEARCH FAILED")

        # ====================================================
        # 2. RAG (Safe Mode)
        # ====================================================

        rag_context = ""

        if (
            RAG_AVAILABLE
            and use_rag
            and not context_text
            and rag is not None
        ):

            try:

                rag_docs = rag.get_context(final_question, top_k=4)

                if rag_docs:
                    rag_context = rag_docs
                    print("RAG SEARCH SUCCESS")

            except Exception as e:

                print(f"RAG SEARCH ERROR: {str(e)}")

        # ====================================================
        # 3. BUILD SYSTEM PROMPT
        # ====================================================

        system_prompt = build_system_prompt(web_context)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # ====================================================
        # 4. HISTORY
        # ====================================================

        if chat_history:

            recent_history = chat_history[-MAX_HISTORY_MESSAGES:]

            for message in recent_history:

                if not isinstance(message, dict):
                    continue

                role = message.get("role")
                content = message.get("content")

                if role not in ["user", "assistant"]:
                    continue

                if not content:
                    continue

                messages.append({
                    "role": role,
                    "content": str(content)
                })

        # ====================================================
        # 5. BUILD USER QUERY
        # ====================================================

        final_prompt = final_question

        if context_text:

            final_prompt = f"""
The user provided the following document context.

---------------- DOCUMENT ----------------
{str(context_text)[:12000]}
-------------- END DOCUMENT --------------

User's request:
{final_question}
"""

        elif rag_context:

            final_prompt = f"""
The following information was retrieved from the user's previously uploaded documents.

---------------- DOCUMENT CONTEXT ----------------
{rag_context}
-------------- END DOCUMENT CONTEXT --------------

User's request:
{final_question}
"""

        messages.append({
            "role": "user",
            "content": final_prompt
        })

        # ====================================================
        # 6. GROQ
        # ====================================================

        answer, error = _ask_groq(messages)

        if error:
            return f"Zenthor could not respond. Error: {error}"

        return clean_response(answer)

    except Exception as e:

        return f"An error occurred in Zenthor: {str(e)}"


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_file):

    try:

        import pypdf

        reader = pypdf.PdfReader(pdf_file)
        text_parts = []

        for page in reader.pages:

            try:

                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)

            except Exception:
                continue

        return "\n\n".join(text_parts).strip()

    except Exception as e:

        return f"PDF read error: {str(e)}"


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def ask_ai_tutor(
    user_question,
    chat_history=None,
    grade=None,
    subject=None,
    context_text=None,
    image_file=None,
):

    extra_context = ""

    if grade:
        extra_context += f"\nUser level/context: {grade}"
    if subject:
        extra_context += f"\nSubject/context: {subject}"
    if context_text:
        extra_context += "\n\nDocument context:\n" + context_text

    return ask_ai(

        user_question=user_question,
        chat_history=chat_history,

        context_text=(
            extra_context.strip()
            if extra_context
            else None
        ),

    )


# ============================================================
# REFRESH MODELS
# ============================================================

def refresh_models():

    global _available_models
    _available_models = None
    return get_available_models()
