import os
from dotenv import load_dotenv
from chroma_db import create_vector_store

from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = create_vector_store()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="text-generation",
    temperature=0.2,
    max_new_tokens=300,
)

model = ChatHuggingFace(llm=llm)

def chat_with_rag(user_id, query, history=None):

    if history is None:
        history = []

    formatted_history = []

    for msg in history:

        if msg.role == "user":
            formatted_history.append(
                HumanMessage(content=msg.content)
            )

        elif msg.role == "assistant":
            formatted_history.append(
                AIMessage(content=msg.content)
            )

    if formatted_history:

        rewrite_messages = [
            SystemMessage(
                content="""
                Rewrite the user's latest question into a standalone search query.

                Rules:
                - Keep original meaning
                - Include relevant context from chat history
                - Return ONLY the rewritten query
                """
            )
        ] + formatted_history + [
            HumanMessage(content=query)
        ]

        try:
            rewrite_result = model.invoke(rewrite_messages)
            search_question = rewrite_result.content.strip()

        except:
            search_question = query

    else:
        search_question = query

    retriever = db.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "user_id": user_id
            }
        }
    )

    relevant_docs = retriever.invoke(search_question)

    if len(relevant_docs) == 0:
        context = "No relevant tasks found."
    else:
        context = "\n\n".join([
            doc.page_content
            for doc in relevant_docs
        ])

    conversation_history = "\n".join([
        f"{msg.role}: {msg.content}"
        for msg in history
    ])

    system_prompt = """
    You are a smart AI task assistant.

    Your responsibilities:
    - Help users manage tasks
    - Suggest priorities
    - Answer based on task context
    - Continue conversations naturally

    Rules:
    - ONLY use provided task context
    - Do not invent fake tasks
    - Keep responses concise and practical
    """

    human_prompt = f"""
    Conversation History:
    {conversation_history}

    Relevant Tasks:
    {context}

    Current User Question:
    {query}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    try:
        result = model.invoke(messages)
        return result.content

    except Exception as e:
        return f"Error: {str(e)}"