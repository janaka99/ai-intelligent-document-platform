from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship

from app.db.database import Base, get_async_session

class User(SQLAlchemyBaseUserTableUUID, Base):
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)
