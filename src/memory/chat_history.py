from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class ChatHistory:
    store = {}

    @classmethod
    def get_session_history(cls, session_id: str, config: dict = None) -> BaseChatMessageHistory:
        if session_id not in cls.store:
            cls.store[session_id] = ChatMessageHistory()
        return cls.store[session_id]

    @classmethod
    def clear_history(cls, session_id: str):
        if session_id in cls.store:
            del cls.store[session_id]
