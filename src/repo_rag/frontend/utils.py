from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

from repo_rag.components.loader import Loader
from repo_rag.components.vectorstore import Vectorstore


def add_to_vector_store(repo_url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, separators=['\n', ' ', ''])

    documents = Loader.load_and_split(repo_url, text_splitter)

    store = Vectorstore(1)

    batch_size = 900
    total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)

    for i in range(total_batches):
        batch_documents = documents[i * batch_size : (i + 1) * batch_size]
        store.add_docs(docs=batch_documents, batch_size=batch_size)
        progress = (i + 1) / total_batches * 100
        st.session_state['progress'] = progress
        st.session_state['batch_number'] = i + 1
        st.session_state['total_batches'] = total_batches
        st.session_state['repo_url'] = repo_url
        st.rerun()

    st.session_state['vectorstore_created'] = True


def show_progress():
    if 'progress' in st.session_state:
        progress = st.session_state['progress']
        batch_number = st.session_state.get('batch_number', 0)
        total_batches = st.session_state.get('total_batches', 1)
        st.write(f'Batch {batch_number} of {total_batches} processed. ({progress:.2f}% complete)')
        st.progress(progress)
