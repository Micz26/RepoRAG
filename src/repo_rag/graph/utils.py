from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langgraph.graph.state import CompiledStateGraph

from repo_rag.components.prompts import (
    GENERAL_PROMPT_TEMPLATE,
    RETRIEVAL_INSTRUCTION_PROMPT,
    SYSTEM_PROMPT,
)


async def format_docs(docs):
    """Format context and sources for LLM"""
    content = '\n\n\n'.join([d.page_content for d in docs])

    sources = '\n\n\n'.join({f'{doc.metadata["file_name"]} - {doc.metadata["full_url"]}' for doc in docs})

    return content, sources


def print_messages(messages):
    for message in messages:
        print(message.pretty_print())


def format_prompt(query, history=None, retrieved_context='', retrieved_sources=''):
    """
    Returns FORMATTED messages (not a template) using the inputs.
    """
    if history is None:
        history = [BaseMessage(content='')]

    human_template = HumanMessagePromptTemplate.from_template(
        GENERAL_PROMPT_TEMPLATE,
        input_variables=['retrieved_context', 'retrieved_sources', 'query'],
    )

    messages = []

    if retrieved_context.strip():
        messages.append(SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))
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
    """Clear memory of the graph"""
    config = {'configurable': {'thread_id': thread_id}}

    try:
        messages = graph.get_state(config).values['messages']

        for message in messages:
            await graph.aupdate_state(config, {'messages': RemoveMessage(id=message.id)})
    except Exception as e:
        raise RuntimeError('Error occurred while clearing graph memory') from e


async def run_graph(graph: CompiledStateGraph, query, thread_id: str):
    """Compile and invoke graph"""
    config = {'configurable': {'thread_id': thread_id}}

    final_state = await graph.ainvoke(
        {'messages': query},
        config=config,
    )

    return final_state
