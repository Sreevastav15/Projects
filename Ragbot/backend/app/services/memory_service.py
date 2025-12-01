# app/services/memory_service.py

from sqlalchemy.orm import Session
from app.models import ChatSession, ChatMessage
from datetime import datetime, timezone
from langchain_groq import ChatGroq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SUMMARIZE_AFTER_MESSAGES = 5      # When the total messages exceed this → summarize older ones
KEEP_RECENT_MESSAGES = 5           # Keep last N messages intact
MAX_SUMMARY_CHARS = 2500           # safety in case model fails


# --- Token Estimator ---
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)   # rough estimate
    


# --- Retrieve or Create a Chat Session ---
def get_or_create_session(db: Session, document_id: int = None, title: str = "") -> ChatSession:
    # if a session already exists for that document, reuse it
    if document_id:
        session = db.query(ChatSession).filter(ChatSession.document_id == document_id).first()
        if session:
            return session

    # otherwise create a new session
    session = ChatSession(document_id=document_id, title=title or "")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session



# --- Add Message to Session ---
def append_message(db: Session, session_id: int, role: str, content: str) -> ChatMessage:
    token_est = estimate_tokens(content)

    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        token_estimate=token_est
    )

    db.add(msg)
    db.commit()
    db.refresh(msg)

    # check if we need to summarize
    maybe_summarize_and_prune(db, session_id)

    return msg



# --- Get All Messages in a Session ---
def get_recent_messages(db: Session, session_id: int):
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return [{"role": m.role, "content": m.content} for m in msgs]



# --- Summarize + Delete Old Messages ---
def maybe_summarize_and_prune(db: Session, session_id: int):
    """
    If messages exceed SUMMARIZE_AFTER_MESSAGES:
    1. Summarize old ones
    2. Store the summary in ChatSession.last_summary
    3. Delete the old messages
    4. Keep last KEEP_RECENT_MESSAGES untouched
    """

    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    if len(msgs) <= SUMMARIZE_AFTER_MESSAGES:
        return

    # split messages
    old_msgs = msgs[:-KEEP_RECENT_MESSAGES]
    recent_msgs = msgs[-KEEP_RECENT_MESSAGES:]

    # format old conversation text
    text_to_summarize = "\n".join(
        [f"{m.role.upper()}: {m.content}" for m in old_msgs]
    )

    if not text_to_summarize.strip():
        return

    # Summarize using LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile"
    )

    messages = [
        {
            "role": "system",
            "content": "Summarize the following conversation concisely. Preserve key facts and decisions."
        },
        {"role": "user", "content": text_to_summarize}
    ]

    try:
        summary_resp = llm.invoke(messages)
        summary = summary_resp.content
    except:
        summary = text_to_summarize[:MAX_SUMMARY_CHARS]  # fallback

    # Append summary to the session
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if session.last_summary:
        session.last_summary = (session.last_summary + "\n" + summary).strip()
    else:
        session.last_summary = summary

    session.last_summary_at = datetime.now(timezone.utc)

    db.add(session)

    # Delete old messages
    for m in old_msgs:
        db.delete(m)

    db.commit()
