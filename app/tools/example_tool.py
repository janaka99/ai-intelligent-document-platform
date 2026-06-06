from langchain_core.tools import tool


@tool
def get_word_count(text: str) -> dict:
    """Count the number of words and charatcers in a given text ."""
    words = len(text.split())
    characters = len(text)
    return {
        "words": words,
        "characters": characters,
        "message": f"The text has {words} words and {characters} characters",
    }

@tool 
def get_current_time() -> dict:
    """Get the current UTC time."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return {
        "utc_time" : now.isoformat(),
        "readable" : now.strftime("%Y-%m-%d %H:%M:%S UTC")
    }

EXAMPLE_TOOLS = [get_word_count, get_current_time]