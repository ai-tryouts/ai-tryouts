import streamlit as st # Import Streamlit for creating the web app
from datasets import load_dataset # Import Hugging Face's datasets library to load Wikipedia data
from sentence_transformers import SentenceTransformer # Import SentenceTransformer for creating text embeddings
import numpy as np # Import NumPy for numerical operations, especially for handling embeddings
import time # Import time module to measure query duration
from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity for comparing embeddings
import os # Import os module for checking file existence

# Configure the Streamlit page settings
st.set_page_config(page_title="Wikipedia Search", layout="wide") # Set page title and layout

# Display the main title of the application
st.title("🔍 Wikipedia Semantic Search Engine")
# Display a short description of the application
st.write("This app loads Wikipedia articles, encodes them, allows saving/loading embeddings, and performs semantic search with detailed calculation steps.")

# --- Session State Initialization ---
# Check if 'texts' is already in the session state, if not, initialize to None
if 'texts' not in st.session_state:
    st.session_state.texts = None # Holds the loaded article texts
# Check if 'embeddings' is already in the session state, if not, initialize to None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None # Holds the generated embeddings for the articles
# Check if 'model' is already in the session state, if not, load the SentenceTransformer model
if 'model' not in st.session_state:
    with st.spinner("Loading sentence transformer model..."): # Show a spinner while loading
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2') # Load the pre-trained model
    st.success("Model loaded!") # Display a success message once the model is loaded

# --- Step 1: Load Dataset & Embeddings ---
st.subheader("📚 Step 1: Load Dataset & Embeddings") # Display a subheader for this section

# Try to load existing texts and embeddings from local .npy files first
if os.path.exists("wikipedia_texts.npy") and os.path.exists("wikipedia_embeddings.npy"): # Check if both files exist
    if st.session_state.texts is None or st.session_state.embeddings is None: # Load only if not already in session state
        with st.spinner("Loading existing texts and embeddings..."): # Show a spinner during loading
            st.session_state.texts = np.load("wikipedia_texts.npy", allow_pickle=True).tolist() # Load texts and convert to list
            st.session_state.embeddings = np.load("wikipedia_embeddings.npy", allow_pickle=True) # Load embeddings
            # Display a success message with the number of loaded articles
            st.success(f"Loaded {len(st.session_state.texts)} articles and their embeddings from local files!")
else:
    # If local files are not found, inform the user
    st.info("No local embeddings found. Please generate them below or ensure 'wikipedia_texts.npy' and 'wikipedia_embeddings.npy' are present.")

# If texts are still None (i.e., not loaded from local files), fetch them from the Hugging Face dataset
if st.session_state.texts is None: 
    with st.spinner("Loading Wikipedia articles from Hugging Face... (This may take a while on first run)"): # Show spinner
        # Load the 'wikipedia' dataset (20220301.en version), taking the first 1% of the training split
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]", trust_remote_code=True)
        # Extract the text from the first 100 articles, truncating each to 1000 characters for speed
        st.session_state.texts = [item['text'][:1000] for item in dataset.select(range(100))] 
    st.success(f"Loaded {len(st.session_state.texts)} fresh articles from Hugging Face!") # Display success message

# If texts are loaded, provide an option to preview the first 3 articles
if st.session_state.texts and st.checkbox("🔎 Preview Loaded Articles (first 3)", key="preview_loaded"):
    st.write(st.session_state.texts[:3]) # Display the first 3 texts

# --- Step 2: Generate/Verify Embeddings ---
st.subheader("🧠 Step 2: Generate & Save Embeddings (if needed)") # Display subheader for embedding generation

# Create a button to trigger embedding generation and saving
if st.button("⚡ Generate and Save Embeddings Now"):
    if st.session_state.texts is None: # Check if texts are loaded
        st.error("Please load articles first (should happen automatically or via Step 1).") # Show error if no texts
    else:
        with st.spinner("Generating embeddings... This might take a moment."): # Show spinner during generation
            # Encode the loaded texts into embeddings using the pre-loaded model
            st.session_state.embeddings = st.session_state.model.encode(st.session_state.texts, show_progress_bar=True)
            np.save("wikipedia_embeddings.npy", st.session_state.embeddings) # Save embeddings to a .npy file
            # Save the corresponding texts to a .npy file for consistency
            np.save("wikipedia_texts.npy", np.array(st.session_state.texts, dtype=object)) 
            # Display success message
            st.success(f"✅ Embeddings and texts for {len(st.session_state.texts)} articles saved to `wikipedia_embeddings.npy` and `wikipedia_texts.npy`")

# If embeddings are available in session state, display information about them
if st.session_state.embeddings is not None:
    st.write(f"Embeddings for {st.session_state.embeddings.shape[0]} articles are loaded/generated.") # Show count
    if st.checkbox("🔎 Preview first 2 document embeddings", key="preview_embeddings"): # Option to preview embeddings
        st.write(st.session_state.embeddings[:2]) # Display the first 2 embeddings
else:
    # If embeddings are not loaded, show a warning
    st.warning("Embeddings have not been generated or loaded yet. Please use the button above or ensure files exist.")


# --- Step 3: Search Similar Articles ---
st.subheader("🔍 Step 3: Search Similar Articles & View Calculations") # Display subheader for search section

# Proceed only if both texts and embeddings are available
if st.session_state.embeddings is not None and st.session_state.texts is not None:
    query = st.text_input("Enter your search query:", key="search_query") # Create a text input field for the search query
    
    if st.button("🚀 Search", key="search_button"): # Create a search button
        if query: # Proceed if the query is not empty
            st.markdown("---") # Add a horizontal rule
            st.subheader("🧮 Search Calculation Steps:") # Subheader for calculation details
            
            start_time = time.time() # Record the start time for query duration measurement

            # 1. Query Encoding
            st.markdown("**1. Query Encoding:**")
            st.markdown(f"Original Query: `{query}`")
            # Encode the user's query into an embedding using the loaded model
            query_embedding = st.session_state.model.encode([query]) # Note: model.encode expects a list of texts
            with st.expander("🔍 View Query Embedding (Numerical Vector)"):
                st.write(query_embedding) # Display the numerical query embedding
            st.markdown("---") # Add a horizontal rule

            # 2. Cosine Similarity Calculation
            st.markdown("**2. Cosine Similarity Calculation:**")
            st.markdown("The similarity between the query embedding and each document embedding is calculated using Cosine Similarity.")
            st.latex(r'''
            \text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
            ''')
            
            # Calculate cosine similarities between the query embedding and all article embeddings
            # similarities will be a 2D array, e.g., [[score1, score2, ...]]
            similarities = cosine_similarity(query_embedding, st.session_state.embeddings)
            
            with st.expander("🔬 View All Similarity Scores (Query vs. All Documents)"):
                st.write(similarities[0]) # Display the array of all similarity scores
            st.markdown("---") # Add a horizontal rule

            # 3. Ranking
            st.markdown("**3. Ranking Results:**")
            st.markdown("The similarity scores are sorted in descending order to find the most similar documents.")
            # Get the indices of the top 5 most similar articles
            # np.argsort returns indices that would sort an array in ascending order
            # Slicing `[-5:]` gets the last 5 (highest similarity), `[::-1]` reverses to descending order
            top_n_indices = np.argsort(similarities[0])[-5:][::-1] 
            st.markdown(f"Indices of top 5 articles (before mapping to text): `{top_n_indices.tolist()}`")
            
            end_time = time.time() # Record the end time
            query_duration = end_time - start_time # Calculate the query duration
            st.markdown("---") # Add a horizontal rule
            st.success(f"Search and Ranking completed in {query_duration:.4f} seconds.") # Display search time
            st.markdown("---") # Add a horizontal rule
            st.subheader("🏆 Top 5 Most Similar Articles:") # Subheader for results

            # Loop through the top N indices and display each result with detailed calculations
            for i, idx in enumerate(top_n_indices):
                doc_embedding = st.session_state.embeddings[idx] # Get the specific document embedding
                
                # Manual calculation for display (sklearn's cosine_similarity is optimized)
                dot_product = np.dot(query_embedding[0], doc_embedding)
                norm_query = np.linalg.norm(query_embedding[0])
                norm_doc = np.linalg.norm(doc_embedding)
                manual_similarity = dot_product / (norm_query * norm_doc)

                st.markdown(f"**Rank {i+1} (Similarity Score: {similarities[0][idx]:.4f})**")
                st.markdown(f"> {st.session_state.texts[idx][:500]}...") # Preview article text
                
                with st.expander(f"🔍 View Calculation Details & Embedding for Rank {i+1}"):
                    st.markdown("**Calculation Components:**")
                    st.markdown(f"- Query Embedding (A) - (Shape: {query_embedding[0].shape}) - *See above for full vector*")
                    st.markdown(f"- Document Embedding for this article (B) - (Shape: {doc_embedding.shape})")
                    st.code(f"{doc_embedding}", language=None)
                    st.markdown(f"- Dot Product (A ⋅ B): `{dot_product:.4f}`")
                    st.markdown(f"- Magnitude of Query Embedding (||A||): `{norm_query:.4f}`")
                    st.markdown(f"- Magnitude of Document Embedding (||B||): `{norm_doc:.4f}`")
                    st.markdown(f"- Calculated Similarity ((A ⋅ B) / (||A|| ⋅ ||B||)): `{manual_similarity:.4f}` (matches score)")
                st.markdown("---") # Add a horizontal rule after each result
        else:
            st.warning("Please enter a search query.") # Show warning if query is empty
else:
    # Show warning if texts or embeddings are not available for searching
    st.warning("Please load/generate texts and embeddings (Steps 1 & 2) before searching.")

# Footer section
st.markdown("---") # Add a horizontal rule
# Display a "Made with" message crediting the libraries used
st.markdown("Made with ❤️ using [Sentence Transformers](https://www.sbert.net/), [Hugging Face Datasets](https://huggingface.co/docs/datasets), [Scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)")
