# Phase 3: Memory Management Module

**Phase 3 is complete!** Here is an explanation of the Memory Manager and the three-layer strategy we built to protect the LLM context window.

### 1. Token Counter (`app/core/token_counter.py`)
We built a strict token counting function using `tiktoken`. 
- **Tokens vs Words:** LLMs don't read words, they read tokens. If we simply counted "words" or "characters" to prevent the LLM from crashing, we would eventually miscalculate and crash the system. Using `tiktoken` ensures we perfectly align with OpenAI's strict context limits.

### 2. Three-Layer Memory Strategy (`app/memory/manager.py`)
The `MemoryManager.build_context()` method is the heart of the chat's persistence. It guarantees the LLM remembers important context without hallucinating or breaking the token budget. 

It does this in three defensive layers:

#### Layer 1: Sliding Window
Instead of injecting the *entire* conversation history (which becomes massively expensive and causes the LLM to "forget" things in the middle of long prompts), we grab only the last `WINDOW_SIZE` (8) messages. This ensures the LLM has word-for-word accuracy of the immediate conversation context.

#### Layer 2: Rolling Summary
What happens to the 9th message? If the history crosses our `SUMMARIZE_THRESHOLD`, the `maybe_summarize()` function wakes up. 
It grabs any old messages that fell out of the window and sends them to the LLM behind the scenes with the prompt: *"Compress the following conversation into a concise summary."* 
This compressed summary is prepended to the final prompt as a `SystemMessage`, ensuring the LLM vaguely remembers what was talked about an hour ago without paying the token cost of verbatim history!

#### Layer 3: Token Guard
Before returning the final context list to the agent, we count the tokens of the Summary + Sliding Window. If the user sent a massive block of text and we exceed the `TOKEN_BUDGET` (12,000), a `while` loop aggressively trims the oldest messages from the Sliding Window until it's safe to send.

### Why does this matter?
Without this strategy, you would just send `history.append(new_message)` to OpenAI on every turn. Eventually, a single API request would contain 50 pages of text, cost $2.00 per message, and then crash your server with a `400 Token Limit Exceeded` error!
