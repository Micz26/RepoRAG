from langchain_openai import ChatOpenAI

from repo_rag.components.constants import OPEN_AI_API_KEY
from repo_rag.components.models import ExpandedQuery

chat_llm = ChatOpenAI(model='gpt-4o-mini', api_key=OPEN_AI_API_KEY, temperature=0.5, max_tokens=5000)

query_expansion_llm = chat_llm.with_structured_output(ExpandedQuery)
