# backend/routes/answer.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import SessionLocal
from app.models import Document, Answer
from app.services.qa_service import get_answer
from app.services.memory_service import (
    get_or_create_session,
    append_message,
    get_recent_messages
)

router = APIRouter(prefix="/answer", tags=["Answer"])

class AnswerRequest(BaseModel):
    question: str
    document_id: int


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/")
async def answer_question(payload: AnswerRequest, db: Session = Depends(get_db)):

    doc = db.query(Document).filter(Document.id == payload.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    session = get_or_create_session(
        db,
        document_id=doc.id,
        title=doc.filename
    )

    append_message(db, session.id, "user", payload.question)

    recent_messages = get_recent_messages(db, session.id)

    chat_history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in recent_messages
    ]

    full_history = []
    if session.last_summary:
        full_history.append(("system", f"Conversation summary: {session.last_summary}"))
    full_history.extend(chat_history)

    answer = get_answer(
        question=payload.question,
        vector_path=doc.vector_path,
        chat_history=full_history,
        summary=session.last_summary 
    )

    append_message(db, session.id, "assistant", answer)

    answer_entry = Answer(
        document_id= doc.id,
        question_text= payload.question,
        answer_text= answer
    )
    db.add(answer_entry)
    db.commit()

    return {
        "question": payload.question,
        "answer": answer,
        "session_id": session.id
    }
