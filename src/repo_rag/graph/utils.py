from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langgraph.graph.state import CompiledStateGraph
from langchain_core.documents import Document

from repo_rag.components.prompts import (
    GENERAL_PROMPT_TEMPLATE,
    RETRIEVAL_INSTRUCTION_PROMPT,
    SYSTEM_PROMPT,
)
from repo_rag.graph.state import RepoConvoState


async def format_docs(docs: list[Document]) -> tuple[str]:
    """
    Format context and sources for LLM

    Parameters
    ----------
    docs : list[Document]
        Retrieved docuemnts

    Returns
    -------
    tuple[str]
        Retreieved context as str and sources as str
    """
    content = '\n\n\n'.join([d.page_content for d in docs])

    sources = '\n\n\n'.join({f'{doc.metadata["file_name"]} - {doc.metadata["full_url"]}' for doc in docs})

    return content, sources


def print_messages(messages):
    for message in messages:
        print(message.pretty_print())


def format_prompt(query, history=None, retrieved_context: str = '', retrieved_sources: str = ''):
    """
    Returns formatted messages (not a template) using the inputs.

    Parameters
    ----------
    query : BaseMessage
        User's query message.
    history : list of BaseMessage, optional
        Conversation history, by default None.
    retrieved_context : str, optional
        Retrieved context from repository vectorstore, by default ''.
    retrieved_sources : str, optional
        Sources of the retrieved context, by default ''.

    Returns
    -------
    list of BaseMessage
        A list of formatted chat messages.
    """
    if history is None:
        history = [BaseMessage(content='')]

    human_template = HumanMessagePromptTemplate.from_template(
        GENERAL_PROMPT_TEMPLATE,
        input_variables=['retrieved_context', 'retrieved_sources', 'query'],
    )

    messages = []

    if retrieved_context.strip():
        messages.append(
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
        )  # Dont include RETRIEVAL_INSTRUCTION_PROMPT again if
    else:
        messages.append(SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT + '\n' + RETRIEVAL_INSTRUCTION_PROMPT))

    messages.extend(history)
    messages.append(human_template)

    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template.format_messages(
        retrieved_context=retrieved_context,
        retrieved_sources=retrieved_sources,
        query=query.content,
    )


async def clear_memory(graph: CompiledStateGraph, thread_id: str) -> None:
    """
    Clear memory of the graph

    Parameters
    ----------
    graph : CompiledStateGraph
        complied graph
    thread_id : str
        thread_id for checkpointer

    Raises
    ------
    RuntimeError
        RuntimeError
    """
    config = {'configurable': {'thread_id': thread_id}}

    try:
        messages = graph.get_state(config).values['messages']

        for message in messages:
            await graph.aupdate_state(config, {'messages': RemoveMessage(id=message.id)})
    except Exception as e:
        raise RuntimeError('Error occurred while clearing graph memory') from e


async def run_graph(graph: CompiledStateGraph, query: str, thread_id: str) -> RepoConvoState:
    """
    Compile and invoke graph

    Parameters
    ----------
    graph : CompiledStateGraph
        complied graph
    query : str
        query
    thread_id : str
        thread_id for checkpointer

    Returns
    -------
    RepoConvoState
        Final state of conversation
    """
    config = {'configurable': {'thread_id': thread_id}}

    final_state = await graph.ainvoke(
        {'messages': query},
        config=config,
    )

    return final_state
