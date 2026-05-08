from fastapi import FastAPI
from chroma_db import create_vector_store
from chat import chat_with_rag
from pydantic import BaseModel
from typing import Literal

app = FastAPI()

class TaskRequest(BaseModel):
    userId: str
    taskId: str
    title: str
    content: str
    is_important: bool
    taskDeadline: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    userId: str
    query: str
    history: list[Message]

vectorstore = create_vector_store()

@app.get("/health")
def index():
    return {"message": "API Chatbot đang chạy"}

@app.post("/insert-task")
def insert_task(task: TaskRequest):

    text = f"""
    Title: {task.title}
    Description: {task.content}
    Is important: {task.is_important}
    Deadline: {task.taskDeadline}
    """

    vectorstore.add_texts(
        texts=[text],
        metadatas=[{
            "user_id": task.userId,
            "is_important": task.is_important,
            "deadline": task.taskDeadline
        }],
        ids=[task.taskId]
    )   

    return {"message": "ok"}

@app.delete("/delete-task/{task_id}")
def delete_task(task_id: str):

    vectorstore.delete(ids=[task_id])

    return {
        "message": "ok"
    }

@app.post("/chat")
def chat(req: ChatRequest):

    response = chat_with_rag(
        req.userId,
        req.query,
        req.history
    )

    return {
        "response": response
    }