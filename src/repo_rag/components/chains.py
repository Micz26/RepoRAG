from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate


from repo_rag.components.constants import OPEN_AI_API_KEY
from repo_rag.components.prompts import QUERY_EXPANSION_SYSTEM_PROMPT
from repo_rag.components.llms import query_expansion_llm, chat_llm
from repo_rag.components.prompts import QUERY_EXTRACTION_SYSTEM_PROMPT


prompt = ChatPromptTemplate.from_messages(
    [
        ('system', QUERY_EXTRACTION_SYSTEM_PROMPT),
        ('human', '{question}'),
    ]
)

query_extraction_chain = prompt | chat_llm


prompt = ChatPromptTemplate.from_messages(
    [
        ('system', QUERY_EXPANSION_SYSTEM_PROMPT),
        ('human', '{question}'),
    ]
)
query_expansion_chain = prompt | query_expansion_llm
