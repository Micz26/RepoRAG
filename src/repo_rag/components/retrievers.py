from repo_rag.components.vectorstore import Vectorstore


vectorstore = Vectorstore(1)
vectorstore.create()
vectorstore = vectorstore.load()
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 10})
