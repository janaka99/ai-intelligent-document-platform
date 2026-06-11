from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import get_settings

settings = get_settings()

client = None

def get_mongo_client() -> AsyncIOMotorClient:
    global client
    if client is None:
        client = AsyncIOMotorClient(settings.mongodb_url)
    return client

def get_mongo_db():
    # Use the database specified in the connection string or fallback to "docintel"
    client = get_mongo_client()
    return client.get_default_database("docintel")
