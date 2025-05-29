import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- 1. Generate 20 odd/distinct sentences ---
sentences = [
    "The old lighthouse stood resilient against the crashing waves of the tumultuous sea.",
    "A peculiar melody drifted from the abandoned music box, echoing through the dusty attic.",
    "Cryptic symbols were etched into the ancient stone tablet, puzzling archaeologists for decades.",
    "The aroma of freshly baked bread and strong coffee filled the small, bustling Parisian cafe.",
    "A lone wolf howled at the silver moon, its mournful cry piercing the silent, frozen tundra.",
    "Whispers of forgotten legends and mythical creatures circulated among the village elders.",
    "The intricate gears of the antique clockwork device whirred and clicked with precision.",
    "Sunlight dappled through the dense canopy of the enchanted forest, illuminating hidden paths.",
    "A mysterious traveler, cloaked in shadows, arrived at the inn under the cover of twilight.",
    "The vibrant colors of the coral reef shimmered beneath the crystal-clear turquoise waters.",
    "An eerie silence descended upon the haunted mansion as the grandfather clock struck midnight.",
    "Philosophers debated the nature of reality and consciousness in the hallowed halls of academia.",
    "The scent of rain on dry earth, known as petrichor, is a uniquely refreshing experience.",
    "A flock of starlings painted mesmerizing patterns across the canvas of the evening sky.",
    "The chef meticulously arranged the delicate components of the avant-garde culinary creation.",
    "Ancient ruins, reclaimed by nature, hinted at a civilization lost to the mists of time.",
    "The detective pieced together a series of seemingly unrelated clues to solve the complex case.",
    "A gentle breeze rustled the leaves of the weeping willow, creating a soothing, natural rhythm.",
    "The artist's abstract sculpture challenged conventional perceptions of form and space.",
    "Sparks flew from the blacksmith's hammer as it struck the glowing metal on the anvil."
]

print(f"Generated {len(sentences)} sentences.\n")

# --- 2. Create vector embeddings ---
# Load a pre-trained sentence transformer model
# 'all-MiniLM-L6-v2' is a good starting point: fast and decent quality.
model_name = 'all-MiniLM-L6-v2'
print(f"Loading sentence transformer model: {model_name}...")
model = SentenceTransformer(model_name)

print("Encoding sentences into vector embeddings...")
embeddings = model.encode(sentences)
print(f"Created {embeddings.shape[0]} embeddings, each with dimension {embeddings.shape[1]}.")

print("\n--- Sample of Generated Embeddings (first 5 dimensions) ---")
for i, emb in enumerate(embeddings):
    print(f"Sentence {i:2d}: {str(emb[:5])}...")
print("\n")

# FAISS expects float32 type for embeddings
embeddings = np.array(embeddings).astype('float32')

# --- 3. Store embeddings in a FAISS index with HNSW ---
dimension = embeddings.shape[1]  # Dimension of our embeddings

# HNSW parameters
# M: number of neighbors for each node during graph construction.
# A higher M generally leads to better recall but slower indexing and search.
# Common values are between 16 and 64.
M = 32
# efConstruction: depth of search during index construction.
# A higher efConstruction leads to a better quality index but slower build time.
efConstruction = 200 # Can be increased for better quality, but takes longer to build

# Create an HNSW index
# "HNSW_SQ" uses HNSW with Scalar Quantization for memory efficiency, good for large datasets.
# For smaller datasets or when exactness is paramount and memory isn't an issue, "HNSWFlat" can be used.
# We'll use HNSWFlat here for simplicity and to focus on HNSW.
# The string 'IDMap,HNSW' creates an HNSW index that also maps original IDs.
# We'll use a simpler HNSWFlat index directly.
index_hnsw = faiss.IndexHNSWFlat(dimension, M)
index_hnsw.hnsw.efConstruction = efConstruction # Set construction parameter

print(f"Building HNSW index (M={M}, efConstruction={efConstruction})...")
# Add the embeddings to the index
index_hnsw.add(embeddings)
print(f"Index built. Total vectors in index: {index_hnsw.ntotal}.")

print("\n--- HNSW Graph Structure (Available Information) ---")
# The HNSW object is accessible via index_hnsw.hnsw
hnsw_graph = index_hnsw.hnsw
print(f"Max level in HNSW graph: {hnsw_graph.max_level}")
print(f"Entry point node ID: {hnsw_graph.entry_point}")
print("""
Note on HNSW Graph Introspection:
The FAISS Python API provides limited direct access to the detailed internal graph structure
(like per-node neighbor lists for each layer). The `hnsw_graph.levels` and `hnsw_graph.neighbors`
attributes are SWIG-wrapped C++ objects that can be tricky to parse into Pythonic structures
reliably for all cases, especially for small N where the graph might be degenerate.
A deeper inspection would typically require C++ debugging or specialized FAISS tools.
For this script, we are focusing on the high-level parameters and conceptual understanding.
""")
print("\n")


# --- 4. Illustrate querying the HNSW index ---
# Let's pick a query sentence (can be one of the existing ones or a new one)
query_sentence_index = 0 # Using the first sentence as a query
query_sentence = sentences[query_sentence_index]
# query_sentence = "A strange sound came from the old house." # Example of a new query

print(f"Querying with sentence: \"{query_sentence}\"")

# Encode the query sentence
query_embedding = model.encode([query_sentence]).astype('float32')

# Number of nearest neighbors to retrieve
k = 5  # We want to find the top 5 similar sentences

# Set search-time parameter efSearch
# efSearch: depth of search during querying. Higher is more accurate but slower.
# Should be at least k.
index_hnsw.hnsw.efSearch = 100 # Can be tuned

print(f"Searching for {k} nearest neighbors (efSearch={index_hnsw.hnsw.efSearch})...")
# Perform the search
# D will contain the distances (L2 squared), I will contain the indices of the neighbors
distances, indices = index_hnsw.search(query_embedding, k)

print("\n--- Query Results ---")
print(f"Query: \"{query_sentence}\"")
print(f"\nTop {k} most similar sentences:")
for i in range(k):
    idx = indices[0][i]
    dist = distances[0][i]
    print(f"  {i+1}. Index: {idx}, Distance: {dist:.4f} - \"{sentences[idx]}\"")

print("\n--- Conceptual HNSW Query Steps (More Technical Detail) ---")
print("""
HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor (ANN) search algorithm.
The core idea is to build a multi-layer graph where upper layers provide fast, coarse-grained search and lower layers
provide fine-grained accuracy. FAISS's `IndexHNSWFlat` by default uses L2 (Euclidean) distance.

When querying with a vector `q`, HNSW performs the following conceptual operations:

1.  **Entry Point Identification**:
    - The search begins at a designated entry point node `ep` in the top-most layer (layer `L`).
    - Calculation: The initial `ep` is often fixed or determined during index construction.

2.  **Greedy Search in Upper Layers (Layers `L` down to `1`):**
    - For each layer `lc` from `L` down to `1`:
        a. **Candidate Set Initialization**: The current best node `ep` (from the layer above or initial entry point)
           is the starting point for this layer.
        b. **Iterative Improvement**:
           - Identify all neighbors of `ep` in layer `lc`.
           - For each neighbor `n`:
             - **Calculation**: Compute the distance `dist(q, n)` (e.g., L2 distance) between the query vector `q`
               and the neighbor's vector `n`.
           - **Comparison**: Find the neighbor `n_closest` that has the minimum `dist(q, n_closest)`.
           - **Decision**:
             - If `dist(q, n_closest) < dist(q, ep)`, then `ep` is updated to `n_closest`, and the search continues
               iteratively from this new `ep`.
             - If `dist(q, n_closest) >= dist(q, ep)`, then `ep` is considered a local minimum in layer `lc`.
               The search in this layer stops, and `ep` becomes the entry point for the layer below (`lc-1`).

3.  **Greedy Search in Base Layer (Layer 0):**
    - The node `ep` found in Layer 1 becomes the entry point for Layer 0.
    - Layer 0 contains all the data points.
    - **Candidate List Management**: A dynamic list (often a min-priority queue ordered by distance to `q`) of the
      `efSearch` best candidates found so far is maintained. Let this be `C`.
    - **Iterative Exploration**:
        a. Initialize `C` with the entry point `ep`.
        b. While there are unvisited candidates in `C` that are closer to `q` than the farthest point currently
           in the top-`k` results:
           - Select the closest unvisited candidate `c_current` from `C`. Mark `c_current` as visited.
           - Identify all neighbors of `c_current` in Layer 0.
           - For each neighbor `n_neighbor`:
             - **Calculation**: Compute `dist(q, n_neighbor)`.
             - **Comparison & Update**: If `n_neighbor` has not been visited and `dist(q, n_neighbor)` is
               smaller than the distance of the farthest candidate in `C` (if `C` has `efSearch` elements),
               or if `C` has fewer than `efSearch` elements, then add `n_neighbor` to `C`.
               If `C` exceeds `efSearch` elements, remove the farthest one.
    - The `efSearch` parameter dictates the size of the candidate list explored, balancing search speed and accuracy.
      A larger `efSearch` means more distance calculations and comparisons but a higher chance of finding true NNs.

4.  **Result Extraction**:
    - After the search in Layer 0, the `k` elements in `C` that are closest to `q` (smallest distances) are
      selected and returned as the approximate nearest neighbors.
    - **Calculation**: The distances themselves (e.g., L2 squared) are also returned.

**Key Calculations & Comparisons:**
-   **Distance Calculation**: Repeatedly calculating distances (e.g., `sqrt(sum((v1_i - v2_i)^2))` for L2) between the
    query vector and vectors of nodes in the graph.
-   **Comparison of Distances**: Comparing these calculated distances to make decisions about which node to
    visit next (greedy traversal) or which candidates to keep in the priority queue.

**Note**: This is a conceptual, more detailed overview. FAISS's HNSW is heavily optimized with C++ data structures
and low-level operations. Accessing a live trace of every comparison/calculation is not possible via the Python API.
The `verbose=True` flag on some FAISS indexes provides some high-level stats but not this level of detail.
""")

print("\n--- HNSW Index Details ---")
print(f"Is the index trained? {index_hnsw.is_trained}")
print(f"Number of vectors in index: {index_hnsw.ntotal}")
print(f"Embedding dimension: {index_hnsw.d}")
print(f"HNSW M parameter (used at construction): {M}") # Corrected line
print(f"HNSW efConstruction parameter: {index_hnsw.hnsw.efConstruction}")
print(f"HNSW efSearch parameter (current): {index_hnsw.hnsw.efSearch}")

print("\nScript finished.")

# To run this script:
# 1. Make sure you have Python installed.
# 2. Install the required libraries: pip install sentence-transformers faiss-cpu numpy
# 3. Save this code as a .py file (e.g., vector_search_hnsw.py)
# 4. Run from your terminal: python vector_search_hnsw.py
