# src/memory/chat_history.py
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from src.db.mongo_client import db
from datetime import datetime

collection = db["chat_history"]


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat history backed by MongoDB."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def add_message(self, message: BaseMessage) -> None:
        """Save a message to MongoDB."""
        await collection.insert_one({
            "session_id": self.session_id,
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "timestamp": datetime.utcnow(),
        })

    async def get_messages(self) -> List[BaseMessage]:
        """Load all messages for a session from MongoDB."""
        from langchain_core.messages import messages_from_dict

        cursor = collection.find({"session_id": self.session_id}).sort("timestamp", 1)
        docs = await cursor.to_list(length=1000)
        # Convert to BaseMessage objects
        return messages_from_dict([
            {
                "type": d["type"],
                "data": {
                    "content": d["content"],
                    "additional_kwargs": d.get("additional_kwargs", {}),
                }
            }
            for d in docs
        ])

    async def clear(self) -> None:
        """Delete all messages for a session."""
        await collection.delete_many({"session_id": self.session_id})


class ChatHistory:
    """Factory for MongoDB-backed chat history."""

    @classmethod
    def get_session_history(cls, session_id: str, config: dict = None) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(session_id)
