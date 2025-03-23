from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from repo_rag.graph.nodes import chatbot, fill_template, final_answer, retrieve_data, route_retriever
from repo_rag.graph.state import RepoConvoState
from repo_rag.graph.utils import run_graph


def create_workflow():
    """
    Creates and returns the repo-rag workflow graph.

    Returns:
        StateGraph: The compiled  workflow graph.
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


# if __name__ == '__main__':
#     import asyncio

#     graph = create_workflow()
#     thread_id = '1'

#     while True:
#         query = input('You: ')

#         result = asyncio.run(run_graph(graph, query, thread_id))

#         print('Chat: ' + result['messages'][-1].content)
