from app.models.chat import ChatSession
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

class ChatLimiter:
    def __init__(self):
        # Note: As per the implementation plan, we are keeping the local stack simple 
        # and not requiring Redis. We will enforce turn and token limits strictly.
        pass

    def check_turn_limit(self, session: ChatSession) -> bool:
        """
        Return False if session.turn_count >= MAX_TURNS.
        Why we cap turns: Infinite contexts drain money and crash LLMs.
        It also forces users to start fresh chats, keeping context highly relevant.
        """
        if session.turn_count >= settings.max_turns:
            logger.warning("turn_limit_exceeded", session_id=session.id, turns=session.turn_count)
            return False
        return True

    def check_token_budget(self, estimated_tokens: int) -> bool:
        """
        Return False if the user's initial message is larger than the entire budget!
        """
        if estimated_tokens > settings.token_budget:
            logger.warning("token_budget_exceeded_by_user", tokens=estimated_tokens)
            return False
        return True
