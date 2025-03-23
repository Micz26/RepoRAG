import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

from repo_rag.components.loader import Loader
from repo_rag.components.vectorstore import Vectorstore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_to_vector_store(repo_url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, separators=['\n', ' ', ''])

    documents = Loader.load_and_split(repo_url, text_splitter)

    logger.info(f'Total of {len(documents)} documents')

    store = Vectorstore(1)

    batch_size = 900
    total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)

    st.write(
        f'Processing repository, adding documents to vector store, estimated time {int(total_batches*1.3)} minutes...'
    )

    store.add_docs(docs=documents, batch_size=batch_size)

    st.session_state['repo_url'] = repo_url
    st.session_state['go_to_chatbot'] = True
    st.rerun()
