from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    original_path = Column(String)
    vector_path = Column(String)
    upload_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    questions = relationship("Question", back_populates="document")
    answers = relationship("Answer", back_populates="document")

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    question_text = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="questions")

class Answer(Base):
    __tablename__ = "answers"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    question_text = Column(Text)
    answer_text = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="answers")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable= True)
    title = Column(String, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_summary = Column(Text, nullable=True)
    last_summary_at = Column(DateTime(timezone=True), default = lambda: datetime.now(timezone.utc))

    messages = relationship("ChatMessage", back_populates="session", cascade = "all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role= Column(String(20))
    content= Column(Text)
    created_at= Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    token_estimate= Column(Integer, default=0)

    session = relationship("ChatSession", back_populates="messages")

    
