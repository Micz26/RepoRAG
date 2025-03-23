from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class RepoConvoState(TypedDict):
    """
    Represents the state of the conversation over repository workflow.
    """

    messages: Annotated[list, add_messages]

    should_retrieve: bool
    retrieving_query: str
    retrieved_docs: list[Document]

    prompt: list[BaseMessage]
