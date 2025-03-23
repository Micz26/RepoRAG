from repo_rag.components.vectorstore import Vectorstore


vectorstore = Vectorstore(1)
vectorstore.create()
vectorstore = vectorstore.load()
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 10, 'fetch_k': 20})
