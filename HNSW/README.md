# HNSW Vector Search with FAISS and Sentence Transformers

This project demonstrates how to create vector embeddings for sentences, store them in a FAISS index using the HNSW (Hierarchical Navigable Small World) algorithm, and perform similarity searches.

## `vector_search_hnsw.py`

This Python script performs the following steps:

1.  **Generates Sentences**: A predefined list of 20 distinct sentences is used as the dataset.
2.  **Loads Model**: Loads a pre-trained sentence embedding model (`all-MiniLM-L6-v2`) from the `sentence-transformers` library.
3.  **Creates Embeddings**: Encodes the sentences into dense vector embeddings. A sample of these embeddings (first 5 dimensions) is printed.
4.  **Builds FAISS HNSW Index**:
    *   Initializes a FAISS `IndexHNSWFlat` index. This type of index uses HNSW for approximate nearest neighbor search and stores the full vectors (flat).
    *   The HNSW parameters `M` (number of neighbors per node during construction) and `efConstruction` (depth of search during construction) are set.
    *   The generated embeddings are added to this index.
    *   Basic information about the HNSW graph (max level, entry point) is printed, along with a note about the limitations of detailed graph introspection via the Python API.
5.  **Performs Query**:
    *   A sample query sentence is chosen from the dataset.
    *   The query sentence is encoded into an embedding.
    *   The `efSearch` parameter (depth of search during querying) for HNSW is set.
    *   A search is performed on the index to find the `k` most similar sentences.
6.  **Displays Results**:
    *   The top `k` similar sentences are printed along with their distances to the query.
    *   A detailed conceptual explanation of the HNSW query algorithm steps is printed.
    *   Key details about the configured HNSW index (e.g., `M`, `efConstruction`, `efSearch`) are displayed.

## Prerequisites

*   Python 3.x
*   `pip` (Python package installer)

## Setup and Execution

1.  **Clone or Download**:
    Get the `vector_search_hnsw.py` script onto your local machine.

2.  **Create a Virtual Environment (Recommended)**:
    It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with system-wide Python packages.
    Open your terminal in the directory where you saved the script and run:
    ```bash
    python3 -m venv venv_hnsw
    ```

3.  **Activate the Virtual Environment**:
    *   On macOS and Linux:
        ```bash
        source venv_hnsw/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv_hnsw\Scripts\activate
        ```
    Your terminal prompt should now indicate that you are in the `(venv_hnsw)` environment.

4.  **Install Required Libraries**:
    With the virtual environment activated, install the necessary Python packages:
    ```bash
    pip install sentence-transformers faiss-cpu numpy
    ```
    *   `sentence-transformers`: For generating sentence embeddings.
    *   `faiss-cpu`: Facebook AI Similarity Search library (CPU version). For HNSW indexing and search.
    *   `numpy`: For numerical operations, especially array handling for FAISS.

5.  **Run the Script**:
    Execute the Python script from your terminal (ensure the virtual environment is still active):
    ```bash
    python vector_search_hnsw.py
    ```
    (If `python` is aliased to Python 2 on your system, you might need to use `python3 vector_search_hnsw.py`)

## Expected Output

The script will:
*   Print status messages as it loads the model and processes data.
*   Display a sample of the generated embeddings.
*   Provide some high-level information about the constructed HNSW graph.
*   Show the query sentence and the top 5 most similar sentences found in the dataset, along with their distances.
*   Print a conceptual explanation of the HNSW query process.
*   List key parameters of the HNSW index.

The first time you run the script, `sentence-transformers` will download the `all-MiniLM-L6-v2` model, which might take a few moments depending on your internet connection. Subsequent runs will use the cached model.

## Customization

*   **Sentences**: You can modify the `sentences` list in the script to use your own text data.
*   **Sentence Transformer Model**: Change the `model_name` variable to use a different pre-trained model from `sentence-transformers`.
*   **HNSW Parameters**: Adjust `M`, `efConstruction`, and `efSearch` to see how they affect indexing time, search time, and accuracy.
*   **Query**: Modify `query_sentence_index` or set `query_sentence` to a new string to test different queries.
*   **Number of Neighbors (`k`)**: Change the `k` variable to retrieve a different number of similar items.
