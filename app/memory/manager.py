from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.chat_queries import get_messages, get_active_summary, upsert_summary
from app.core.token_counter import count_tokens
from app.models.chat import ChatMessage, RoleEnum
from app.core.logging import get_logger

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self, window_size: int, summarize_threshold: int, token_budget: int, llm):
        self.window_size = window_size
        self.summarize_threshold = summarize_threshold
        self.token_budget = token_budget
        self.llm = llm

    def _to_langchain_messages(self, db_messages: list[ChatMessage]) -> list[BaseMessage]:
        messages = []
        for m in db_messages:
            if m.role == RoleEnum.user:
                messages.append(HumanMessage(content=m.content))
            else:
                messages.append(AIMessage(content=m.content))
        return messages

    async def maybe_summarize(self, session: AsyncSession, session_id: str, all_messages: list[ChatMessage]):
        """
        If len(all_messages) > SUMMARIZE_THRESHOLD:
          - Take messages outside the window that haven't been summarized
          - Ask the LLM to compress them
          - Upsert into summaries table
        """
        active_summary = await get_active_summary(session, session_id)
        covered_id = active_summary.covers_up_to_message_id if active_summary else None
        
        # We only summarize messages outside the sliding window
        messages_outside_window = all_messages[:-self.window_size]
        
        unsummarized_messages = []
        found_covered = False if covered_id else True
        
        for msg in messages_outside_window:
            if found_covered:
                unsummarized_messages.append(msg)
            if msg.id == covered_id:
                found_covered = True
                
        if not unsummarized_messages:
            return 
            
        current_summary = active_summary.summary_text if active_summary else "No previous summary."
        
        prompt = (
            "Compress the following conversation into a concise summary. "
            "Maintain the key context, facts, and intent.\n\n"
            f"Existing Summary: {current_summary}\n\n"
            "New Messages to compress:\n"
        )
        for msg in unsummarized_messages:
            prompt += f"[{msg.role.value.capitalize()}]: {msg.content}\n"
            
        logger.info("triggering_summarization", session_id=session_id, unsummarized_count=len(unsummarized_messages))
        
        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        new_summary = response.content
        
        last_msg = unsummarized_messages[-1]
        await upsert_summary(session, session_id, new_summary, covers_up_to_message_id=last_msg.id)

    async def build_context(self, session: AsyncSession, session_id: str) -> list[BaseMessage]:
        """
        Returns the list of messages to inject into the prompt.
        """
        # 1. Load all messages
        all_messages = await get_messages(session, session_id)
        
        # 2. Maybe summarize (if history is too long)
        if len(all_messages) > self.summarize_threshold:
            await self.maybe_summarize(session, session_id, all_messages)
            
        # 3. Take window
        window_messages = all_messages[-self.window_size:] if all_messages else []
        langchain_window = self._to_langchain_messages(window_messages)
        
        # 4. Prepend summary
        active_summary = await get_active_summary(session, session_id)
        final_context = []
        if active_summary:
            final_context.append(
                SystemMessage(content=f"Previous conversation summary: {active_summary.summary_text}")
            )
            
        final_context.extend(langchain_window)
        
        # 5. Token guard
        while True:
            current_tokens = count_tokens(final_context)
            if current_tokens <= self.token_budget:
                break
                
            # 6. If over budget, trim oldest from window (but keep summary at index 0 if it exists)
            logger.warning("token_budget_exceeded_trimming", current=current_tokens, budget=self.token_budget)
            if active_summary and len(final_context) > 1:
                final_context.pop(1)
            elif len(final_context) > 0:
                final_context.pop(0)
            else:
                break
                
        return final_context
