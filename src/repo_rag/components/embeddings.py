from langchain_openai import OpenAIEmbeddings

from repo_rag.components.constants import OPEN_AI_API_KEY

openai_embeddings = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
