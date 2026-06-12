import tiktoken
from langchain_core.messages import BaseMessage

def count_tokens(messages: list[BaseMessage] | str, model: str = "gpt-4o") -> int:
    """
    Accurately counts the number of tokens in a string or list of LangChain messages
    using OpenAI's tiktoken library.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base") # Fallback to latest encoding
        
    if isinstance(messages, str):
        return len(encoding.encode(messages))
        
    num_tokens = 0
    for message in messages:
        # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        num_tokens += 3
        # Ensure message content is a string
        content = str(message.content)
        num_tokens += len(encoding.encode(content))
        
    # Every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3 
    return num_tokens
