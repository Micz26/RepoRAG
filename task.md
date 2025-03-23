# Task #1

## Implement a Retrieval-Augmented Generation (RAG) System for a Code Repository

### Requirements

The system should provide a complete pipeline from indexing to RAG, requiring only a GitHub URL as input to build the index.  
The scope of this task is limited to a single repository: [escrcpy](https://github.com/viarotel-org/escrcpy).

Use this reference data for evaluation (please note that this data was generated automatically and may contain errors).

Once the index is built, the user should be able to use the system for question-answering over the repository:

- **Input:** Natural language query (question).
- **Output:** Relevant code locations (files).
- **Quality metric:** Recall@10 over filenames.

### Steps

1. **Start with a working baseline.**
2. **Apply advanced techniques to improve RAG quality,** which may include:
   - Index building algorithm.
   - Query expansion.
   - Reranker.
   - Any other beneficial techniques.

### Additional Requirements

- Provide **clear instructions** on running the prototype.
- Include a script to **run evaluation** for the provided dataset.

### Optional (Bonus Points)

- LLM-generated **answer summaries** for retrieved code.
- **Well-documented** code repository.
- Evaluation of **different latency/quality trade-offs**.
- **Seamless switching** between various LLM and embedding providers (Local/API/Custom).

### Evaluation Criteria

We will assess submissions based on:

1. **Functionality and accuracy** of the RAG system.
2. **Efficiency in retrieval and generation** (e.g., response time, token usage).
3. **Code quality and clarity** of documentation.

> If you're short on time, prioritize **quality and reproducibility**. The key is to provide clear instructions on how to run your code and ensure it works. While documentation and code quality are important, they can be secondary if your solution is solid.
