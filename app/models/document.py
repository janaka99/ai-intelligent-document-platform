import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum
from sqlalchemy import UUID
from sqlalchemy.orm import relationship
import enum

from app.db.database import Base


class DocumentStatus(str, enum.Enum):
    untrained = "untrained"
    chunking = "chunking"
    chunked = "chunked"
    training = "training"
    trained = "trained"


class Document(Base):
    __tablename__ = "documents"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    size = Column(Integer, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    status = Column(
        Enum(DocumentStatus),
        default=DocumentStatus.untrained,
        nullable=False,
    )

    training_progress = Column(
        Integer,
        default=0,
        nullable=False,
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user.id"),
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="documents",
    )