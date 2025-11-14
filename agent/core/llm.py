from langchain_openai import ChatOpenAI
from agent.config import settings

llm = ChatOpenAI(
    model="gpt-4o-mini",  # or whatever
    api_key=settings.openai_api_key,
)