from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document, ChatMessage, ChatSession, Question, Answer

router = APIRouter(prefix="/chathistory", tags=["Chathistory"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/session")
async def load_session(doc_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(404, "Document not found")

    session = (
        db.query(ChatSession).filter(ChatSession.document_id == doc_id).first()
    )
    if not session:
        raise HTTPException(404, "Chat session not found")

    questions = (
        db.query(Question).filter(Question.document_id == doc_id).order_by(Question.created_at.asc()).all()
    )

    question_list = [q.question_text for q in questions]

    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    messages = [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat()
        }
        for m in msgs
    ]

    return {
        "document_id": document.id,
        "session_id": session.id,
        "filename": document.filename,
        "question_list": question_list,
        "summary": session.last_summary or "",
        "messages": messages
    }

@router.get("/full")
async def full_history(doc_id:int, db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.id == doc_id).first()
    if not docs:
        raise HTTPException(404, "Documnet not found")
    
    answers = db.query(Answer).filter(Answer.document_id == doc_id).order_by(Answer.created_at.asc()).all()

    return{
        "doc_id": doc_id,
        "filename": docs.filename,
        "conversation": [{"question": a.question_text, "answer": a.answer_text} for a in answers]
    }



@router.get("/all")
async def all_chats(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    
    return [{"doc_id": d.id, "filename": d.filename} for d in docs]
