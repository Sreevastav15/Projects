from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import SessionLocal
from app.models import Document, Answer, Question, ChatSession
import os, shutil

router = APIRouter(prefix="/delete", tags=["Delete"])

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.delete("/")
async def delete_chat(doc_id: int, db: Session= Depends(get_db)):
    try:
        document= db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        db.query(Question).filter(Question.document_id == doc_id).delete()
        db.query(Answer).filter(Answer.document_id == doc_id).delete()

        sessions = db.query(ChatSession).filter(ChatSession.document_id == doc_id).all()
        for s in sessions:
            db.delete(s)

        vector_path = f"static/chroma_stores/{doc_id}"
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)
        upload_path = f"static/uploads/{doc_id}"
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        db.flush()

        db.delete(document)
        db.commit()

        
    finally:
        db.close()

    return {"message": f"Chat for '{document.filename}' deleted successfully"}