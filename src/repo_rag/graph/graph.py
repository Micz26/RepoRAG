from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from repo_rag.graph.nodes import chatbot, fill_template, final_answer, retrieve_data, route_retriever
from repo_rag.graph.state import RepoConvoState


def create_workflow():
    """
    Creates and returns the repo-rag workflow graph.
    """
    workflow = StateGraph(RepoConvoState)

    workflow.add_node('chatbot', chatbot)
    workflow.add_node('retrieve_data', retrieve_data)
    workflow.add_node('fill_template', fill_template)
    workflow.add_node('final_answer', final_answer)

    workflow.add_edge(START, 'chatbot')
    workflow.add_conditional_edges(
        'chatbot',
        route_retriever,
        {'retrieve_data': 'retrieve_data', END: END},
    )
    workflow.add_edge('retrieve_data', 'fill_template')
    workflow.add_edge('fill_template', 'final_answer')
    workflow.add_edge('final_answer', END)

    memory_saver = MemorySaver()

    graph = workflow.compile(checkpointer=memory_saver)

    return graph
