# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routes import upload, answer, chat_history, delete
import os
from dotenv import load_dotenv

app = FastAPI(title="Document QA Backend")

origins = [
    "http://localhost:5173",  # React frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],       # allow all HTTP methods
    allow_headers=["*"],       # allow all headers
)

Base.metadata.create_all(bind=engine)

app.include_router(upload.router)
app.include_router(answer.router)
app.include_router(chat_history.router)
app.include_router(delete.router)
