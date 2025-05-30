import logging

from langchain_core.messages import AIMessage
from langgraph.graph import END

from repo_rag.components.retrievers import retriever
from repo_rag.components.llms import chat_llm
from repo_rag.components.chains import query_extraction_chain
from repo_rag.components.prompts import route_to_retriever_placeholder
from repo_rag.graph.state import RepoConvoState
from repo_rag.graph.utils import format_prompt, format_docs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def chatbot(state: RepoConvoState):
    """
    Invokes query_extraction chain for retrieval routing.

    Parameters
    ----------
    state : RepoConvoState
        The current workflow state.
    """
    query = state['messages'][-1]
    response = await query_extraction_chain.ainvoke(query)

    logging.info(f'Retriever usage: {response.content}')

    if route_to_retriever_placeholder in response.content:
        response.content = response.content.replace(
            route_to_retriever_placeholder, ''
        )  # removing placeholder from response

        return {**state, 'retrieving_query': query + ' ' + response.content, 'should_retrieve': True}
    else:
        return {**state, 'should_retrieve': False}


async def route_retriever(state: RepoConvoState):
    """
    Decides whether to retrieve docs, or proceed to END

    Parameters
    ----------
    state : RepoConvoState
        The current workflow state.
    """
    should_retrieve = state['should_retrieve']
    if should_retrieve:
        return 'retrieve_data'
    else:
        return 'fill_template'


async def retrieve_data(state: RepoConvoState):
    """
    Retrieves repo documents relevant to the given research question from the vector database.

    Parameters
    ----------
    state : RepoConvoState
        The current workflow state.
    """
    question = state['messages'][-1].content

    retrieved_docs = await retriever.ainvoke(question)

    return {**state, 'retrieved_docs': retrieved_docs}


async def fill_template(state: RepoConvoState):
    """
    Formats the retrieved data into a structured AI prompt.

    Parameters
    ----------
    state : RepoConvoState
        The current workflow state.
    """
    query = state['messages'][-1]

    retrieved_content, retrieved_sources = '', ''

    if state['should_retrieve']:
        retrieved_docs = state['retrieved_docs']
        retrieved_content, retrieved_sources = await format_docs(retrieved_docs)

    prompt = format_prompt(
        query=query,
        retrieved_context=retrieved_content,
        retrieved_sources=retrieved_sources,
        history=state['messages'][:-1],
    )

    return {**state, 'prompt': prompt}


async def final_answer(state: RepoConvoState):
    """
    Generates final answer using the AI model.

    Parameters
    ----------
    state : RepoConvoState
        The current workflow state.
    """
    prompt = state['prompt']
    response = await chat_llm.ainvoke(prompt)

    return {**state, 'messages': [AIMessage(response.content)]}
