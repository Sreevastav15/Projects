from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from app.services.google_embedding import GoogleTextEmbedding
from app.services.reranker import rerank
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def get_answer(question, vector_path, chat_history, summary=None):

    # === Build Vectorstore ===
    embeddings = GoogleTextEmbedding() 
    vectorstore = Chroma(persist_directory=vector_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k":15})

    # === Retrieve relevant document chunks ===
    docs = retriever.invoke(question)
    rerank_docs = rerank(question, docs)[:5]

    context_blocks = "\n\n".join([d.page_content for d in rerank_docs])

    # === Build conversation memory ===
    conversation_memory = ""

    if summary:
        conversation_memory += f"\nSummary of previous conversation:\n{summary}\n"

    for role, content in chat_history[-6:]:
        conversation_memory += f"{role.upper()}: {content}\n"

    # === Build final prompt to send to ChatGroq ===
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="openai/gpt-oss-120b")

    prompt = f"""
        You are an intelligent assistant that answers questions using BOTH:
        1. Previous user conversation memory
        2. Retrieved document chunks

        Conversation Memory:
        {conversation_memory}

        ==========
        Document Chunks:
        {context_blocks}
        ==========

        Rules:
        1. Use BOTH document excerpts and conversation memory.
        2. If question refers to previous conversation, follow memory.
        3. Only say “I am not sure” if BOTH memory and document lack content.
        4. Do NOT use HTML.
        5. Always provide the page number from where the answer was extracted in the document at the bottom of the answer. Do not provide the page number or the source in the answer.
        6. **Do NOT use LaTeX or mathematical equation formatting. 
        Avoid \[ \], \( \), $$ $$, and backslash-based math. 
        Write equations in plain text only.**

        Question: {question}

        Answer in Markdown:
        """

    # === Run the model ===
    result = llm.invoke(prompt)
    final_answer = f"{result.content.strip()}"

    return final_answer
