from typing import List
from abc import ABC, abstractmethod
from typing import Any
from langchain_openai import ChatOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

class BaseAgent(ABC):
    """
    All agents inherit from this.
    Handles LLM setup, logging, and enforces a common interface
    """
    def __init__(self,model_name:str = "gpt-4o-mini", temperature:float = 0.1, tools: List = [], system_prompt:str = "") -> None:
        self.tools = tools
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.settings = get_settings()  
        self.temperature = temperature
        self.llm = self.build_llm()
        self.graph = self._build_graph()
        logger.info(
            "agent_initialized",
            agent=self.__class__.__name__,
            tools=[t.name for t in self.tools],
            model=self.model_name,
        )

    def build_llm(self) -> ChatOpenAI:
        """Build the LLM. Override this to swap providers."""
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=settings.openai_api_key,
        )
        # Bind tools so the LLM knows it can call them
        if self.tools:
            return llm.bind_tools(self.tools)
        return llm


    @abstractmethod
    def _build_graph(self, state):  
        """Every agent must implment this - define the Langgraph Flow."""   
        ...

    async def run(self,user_input:str,thread_id:str = None) -> dict[str,Any]:
        """
        Public method FastAPI routes call.
        Returns a clean dict — never expose raw LangGraph internals.
        """
        logger.info(
            "agent_run_started",
            agent=self.__class__.__name__,
            thread_id=thread_id,
            input_preview=user_input[:80],
        )

        try:
            result = await self.__run(user_input,thread_id)
            logger.info(
                "agent_run_finished",
                agent=self.__class__.__name__,
                thread_id=thread_id,
            )
            return result
        except Exception as e:
            logger.exception(
                "agent_run_failed",
                agent=self.__class__.__name__,
                error=str(e),
            )
            raise
    
    @abstractmethod
    async def _run(self, user_input: str, thread_id: str = None) -> dict[str, Any]:
        """Internal run — override in each agent."""
        ...