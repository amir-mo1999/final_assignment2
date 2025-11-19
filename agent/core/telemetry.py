from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from agent.config.settings import settings

langfuse_client = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)
langfuse_handler = CallbackHandler()

if not langfuse_client.auth_check():
    raise RuntimeError("Langfuse auth failed. Check LANGFUSE_* environment variables.")
