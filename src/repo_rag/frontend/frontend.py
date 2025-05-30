import asyncio

import streamlit as st

from repo_rag.graph.graph import create_workflow
from repo_rag.frontend.utils import add_to_vector_store


def ui():
    """
    GUI main function
    """
    st.title('Repo Rag Assistant')

    if 'repo_url' not in st.session_state:
        st.session_state['repo_url'] = None

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    if 'go_to_chatbot' not in st.session_state:
        st.header('Welcome to Repo Rag Assistant!')
        st.write('Provide a GitHub repository URL to create a vector database.')

        repo_url = st.text_input('Enter GitHub repository URL:')

        if repo_url:
            if st.button('Add Repository to Vector Database'):
                add_to_vector_store(repo_url)
                st.write(f'Repository {repo_url} has been successfully added to the vector store!')

        if st.button('Go to Chatbot'):
            st.session_state['go_to_chatbot'] = True
            st.rerun()

    elif 'go_to_chatbot' in st.session_state and st.session_state['go_to_chatbot']:
        st.header('Chat with Repo Rag Assistant')

        if 'graph' not in st.session_state:
            st.session_state['graph'] = None

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        if not st.session_state['graph']:
            graph = create_workflow()
            st.session_state['graph'] = graph

        thread_id = '1'

        user_input = st.text_input('Ask question about repository:', '')

        if user_input:
            session_graph = st.session_state['graph']
            result = asyncio.run(
                session_graph.ainvoke(
                    {'messages': user_input},
                    config={'configurable': {'thread_id': thread_id}},
                )
            )
            st.session_state['graph'] = session_graph
            chat_response = result['messages'][-1].content

            st.session_state['messages'].append({'role': 'user', 'content': user_input})
            st.session_state['messages'].append({'role': 'chat', 'content': chat_response})

        for message in reversed(st.session_state['messages']):
            if message['role'] == 'user':
                st.markdown(f'**User:** {message["content"]}')
            elif message['role'] == 'chat':
                st.markdown(f'**Repo Rag Assistant:** {message["content"]}')
