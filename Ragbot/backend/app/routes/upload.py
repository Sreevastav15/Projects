# backend/routes/upload.py
from fastapi import APIRouter, UploadFile, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.services.pdf_service import extract_text
from app.services.embedding_service import create_vectorstore
from app.models import Document
import shutil, os

router = APIRouter(prefix="/upload", tags=["Upload"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
async def upload_pdf(file: UploadFile, db: Session = Depends(get_db)):
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = f"{upload_dir}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = extract_text(file_path)
    vector_path = create_vectorstore(chunks, file.filename.split(".")[0])

    doc = Document(filename=file.filename, original_path=file_path, vector_path=vector_path)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {"document_id": doc.id}
