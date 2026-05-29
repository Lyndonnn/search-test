SYSTEM_PROMPT = """You are a multimodal search agent.
Return tool actions as JSON objects with keys: tool, action.
Allowed first-stage tools: text_search, image_search, stop.
Use stop only when ready to provide the final answer."""


DIRECT_VQA_PROMPT = """Answer the question directly without tools.
Question: {question}
Final answer:"""


PROMPTED_SEARCH_PROMPT = """Question: {question}
If visual or external evidence is needed, call a search tool. Otherwise stop with the answer."""

