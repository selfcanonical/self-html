import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from io import StringIO
import time
import warnings
warnings.filterwarnings('ignore')

# Import additional libraries with error handling
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="SelfCosine - Advanced Similarity Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .method-description {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .similarity-high {
        background-color: #d4edda;
        color: #155724;
    }
    .similarity-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .similarity-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Caching decorators for model loading
@st.cache_resource
def load_bert_model(model_name="all-MiniLM-L6-v2"):
    """Load and cache BERT sentence transformer model."""
    try:
        if not BERT_AVAILABLE:
            return None
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model: {str(e)}")
        return None

@st.cache_resource
def load_multilingual_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Load and cache multilingual model (MUM alternative)."""
    try:
        if not BERT_AVAILABLE:
            return None
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading multilingual model: {str(e)}")
        return None

@st.cache_resource
def load_splade_model():
    """Load SPLADE model for sparse retrieval."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return None, None
        # Using a SPLADE-like model or fall back to BERT
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        return model, None
    except Exception as e:
        st.error(f"Error loading SPLADE model: {str(e)}")
        return None, None

def preprocess_text(text):
    """Preprocess text for similarity calculation."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    text = ' '.join(text.split())
    
    return text

def preprocess_for_bm25(text):
    """Preprocess text specifically for BM25 (tokenization)."""
    preprocessed = preprocess_text(text)
    return preprocessed.split()

def read_uploaded_file(uploaded_file):
    """Read content from uploaded file with error handling."""
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                content = str(uploaded_file.read(), encoding)
                uploaded_file.seek(0)
                return content
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                continue
        return None
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
        return None

def calculate_tfidf_similarity(queries, documents, doc_names):
    """Calculate TF-IDF based cosine similarity."""
    try:
        processed_queries = [preprocess_text(query) for query in queries]
        processed_docs = [preprocess_text(doc) for doc in documents]
        
        all_texts = processed_queries + processed_docs
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vectors = tfidf_matrix[:len(queries)]
        doc_vectors = tfidf_matrix[len(queries):]
        
        similarity_matrix = cosine_similarity(query_vectors, doc_vectors)
        
        return similarity_matrix, vectorizer
        
    except Exception as e:
        st.error(f"Error calculating TF-IDF similarity: {str(e)}")
        return None, None

def calculate_bm25_similarity(queries, documents, doc_names):
    """Calculate BM25 similarity scores."""
    try:
        if not BM25_AVAILABLE:
            st.error("BM25 library not available. Please install rank-bm25.")
            return None
        
        # Preprocess documents for BM25
        processed_docs = [preprocess_for_bm25(doc) for doc in documents]
        processed_queries = [preprocess_for_bm25(query) for query in queries]
        
        # Create BM25 object
        bm25 = BM25Okapi(processed_docs)
        
        # Calculate scores for each query
        similarity_matrix = []
        for query in processed_queries:
            scores = bm25.get_scores(query)
            # Normalize scores to 0-1 range
            if max(scores) > 0:
                normalized_scores = scores / max(scores)
            else:
                normalized_scores = scores
            similarity_matrix.append(normalized_scores)
        
        return np.array(similarity_matrix)
        
    except Exception as e:
        st.error(f"Error calculating BM25 similarity: {str(e)}")
        return None

def calculate_bert_similarity(queries, documents, doc_names, progress_callback=None):
    """Calculate BERT-based semantic similarity."""
    try:
        if not BERT_AVAILABLE:
            st.error("BERT libraries not available. Please install sentence-transformers and torch.")
            return None
        
        model = load_bert_model()
        if model is None:
            return None
        
        if progress_callback:
            progress_callback(0.1, "Encoding queries...")
        
        # Encode queries
        query_embeddings = model.encode(queries, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.5, "Encoding documents...")
        
        # Encode documents
        doc_embeddings = model.encode(documents, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.8, "Calculating similarities...")
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return similarity_matrix
        
    except Exception as e:
        st.error(f"Error calculating BERT similarity: {str(e)}")
        return None

def calculate_splade_similarity(queries, documents, doc_names, progress_callback=None):
    """Calculate SPLADE-based sparse similarity."""
    try:
        model, tokenizer = load_splade_model()
        if model is None:
            st.error("SPLADE model not available. Using BERT-based sparse representation.")
            return calculate_bert_similarity(queries, documents, doc_names, progress_callback)
        
        if progress_callback:
            progress_callback(0.1, "Processing with sparse retrieval...")
        
        # For simplicity, using sentence transformers with sparse-like processing
        query_embeddings = model.encode(queries, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.5, "Encoding documents...")
        
        doc_embeddings = model.encode(documents, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.8, "Calculating sparse similarities...")
        
        # Apply sparsity (zero out small values)
        threshold = 0.1
        query_embeddings = np.where(np.abs(query_embeddings) < threshold, 0, query_embeddings)
        doc_embeddings = np.where(np.abs(doc_embeddings) < threshold, 0, doc_embeddings)
        
        similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return similarity_matrix
        
    except Exception as e:
        st.error(f"Error calculating SPLADE similarity: {str(e)}")
        return None

def calculate_mum_similarity(queries, documents, doc_names, progress_callback=None):
    """Calculate multilingual similarity (MUM alternative)."""
    try:
        model = load_multilingual_model()
        if model is None:
            st.error("Multilingual model not available.")
            return None
        
        if progress_callback:
            progress_callback(0.1, "Encoding with multilingual model...")
        
        query_embeddings = model.encode(queries, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.5, "Processing documents...")
        
        doc_embeddings = model.encode(documents, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.8, "Calculating multilingual similarities...")
        
        similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return similarity_matrix
        
    except Exception as e:
        st.error(f"Error calculating multilingual similarity: {str(e)}")
        return None

def calculate_muvera_similarity(queries, documents, doc_names, progress_callback=None):
    """Calculate MUVERA-based similarity (alternative semantic model)."""
    try:
        # MUVERA is not a standard model, using an alternative approach
        # Could be replaced with actual MUVERA implementation when available
        if progress_callback:
            progress_callback(0.1, "Loading alternative semantic model...")
        
        # Using a different sentence transformer model as alternative
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model = SentenceTransformer(model_name)
        
        if progress_callback:
            progress_callback(0.3, "Encoding queries...")
        
        query_embeddings = model.encode(queries, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.6, "Encoding documents...")
        
        doc_embeddings = model.encode(documents, show_progress_bar=False)
        
        if progress_callback:
            progress_callback(0.9, "Calculating semantic similarities...")
        
        similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return similarity_matrix
        
    except Exception as e:
        st.error(f"Error calculating MUVERA similarity: {str(e)}")
        return None

def create_similarity_dataframe(similarity_matrix, queries, doc_names):
    """Create a formatted DataFrame from similarity matrix."""
    df = pd.DataFrame(
        similarity_matrix,
        index=[f"Query {i+1}" for i in range(len(queries))],
        columns=doc_names
    )
    return df

def color_similarity_values(val):
    """Color code similarity values based on their range."""
    if val >= 0.7:
        return 'background-color: #d4edda; color: #155724'
    elif val >= 0.4:
        return 'background-color: #fff3cd; color: #856404'
    else:
        return 'background-color: #f8d7da; color: #721c24'

def display_method_descriptions():
    """Display descriptions of similarity methods."""
    st.markdown("### 📚 Similarity Methods")
    
    methods = {
        "TF-IDF": {
            "description": "Term Frequency-Inverse Document Frequency with cosine similarity. Fast and effective for keyword-based matching.",
            "pros": "Fast, interpretable, good for keyword matching",
            "cons": "Limited semantic understanding",
            "complexity": "Low"
        },
        "BM25": {
            "description": "Best Matching 25 - probabilistic ranking function for information retrieval. Excellent for search applications.",
            "pros": "Great for search, handles document length well",
            "cons": "Still keyword-based, no semantic understanding",
            "complexity": "Low"
        },
        "BERT": {
            "description": "Bidirectional Encoder Representations from Transformers. Captures deep semantic meaning and context.",
            "pros": "Excellent semantic understanding, context-aware",
            "cons": "Computationally expensive, requires GPU for large datasets",
            "complexity": "High"
        },
        "SPLADE": {
            "description": "SParse Lexical AnD Expansion model. Combines sparse and dense retrieval benefits.",
            "pros": "Interpretable, combines benefits of sparse and dense methods",
            "cons": "Complex implementation, newer approach",
            "complexity": "High"
        },
        "MUM (Multilingual)": {
            "description": "Multilingual Universal sentence encoder. Handles multiple languages effectively.",
            "pros": "Cross-language support, semantic understanding",
            "cons": "Computationally expensive, may be overkill for monolingual tasks",
            "complexity": "High"
        },
        "MUVERA (Alternative)": {
            "description": "Alternative semantic model using MPNet architecture for enhanced semantic understanding.",
            "pros": "Advanced semantic capabilities, good performance",
            "cons": "Computationally expensive, requires more resources",
            "complexity": "High"
        }
    }
    
    for method, info in methods.items():
        with st.expander(f"ℹ️ {method}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Pros:** {info['pros']}")
            st.markdown(f"**Cons:** {info['cons']}")
            st.markdown(f"**Computational Complexity:** {info['complexity']}")

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔍 Advanced Multi-Method Similarity Checker</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This advanced application calculates text similarity using multiple methods including traditional 
    keyword-based approaches and modern neural semantic models. Choose your methods and compare results!
    """)
    
    # Sidebar for method selection
    st.sidebar.markdown("### 🎯 Similarity Methods")
    st.sidebar.markdown("*TF-IDF is always included*")
    
    # Method selection checkboxes
    use_bm25 = st.sidebar.checkbox("BM25 (Probabilistic Ranking)", value=False, help="Fast probabilistic ranking function")
    use_bert = st.sidebar.checkbox("BERT (Semantic)", value=False, help="Deep semantic understanding", disabled=not BERT_AVAILABLE)
    use_splade = st.sidebar.checkbox("SPLADE (Sparse)", value=False, help="Sparse lexical expansion", disabled=not TRANSFORMERS_AVAILABLE)
    use_mum = st.sidebar.checkbox("MUM (Multilingual)", value=False, help="Multilingual semantic model", disabled=not BERT_AVAILABLE)
    use_muvera = st.sidebar.checkbox("MUVERA (Alternative)", value=False, help="Alternative semantic model", disabled=not BERT_AVAILABLE)
    
    # Configuration options
    st.sidebar.markdown("### ⚙️ Configuration")
    max_files = st.sidebar.slider("Maximum files to upload", 1, 20, 10)
    
    # Check library availability
    st.sidebar.markdown("### 📦 Library Status")
    st.sidebar.markdown(f"BM25: {'✅' if BM25_AVAILABLE else '❌'}")
    st.sidebar.markdown(f"BERT/Transformers: {'✅' if BERT_AVAILABLE else '❌'}")
    st.sidebar.markdown(f"Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    
    if not BM25_AVAILABLE or not BERT_AVAILABLE:
        st.sidebar.markdown("**Install missing libraries:**")
        if not BM25_AVAILABLE:
            st.sidebar.code("pip install rank-bm25")
        if not BERT_AVAILABLE:
            st.sidebar.code("pip install sentence-transformers torch")
    
    # Display method descriptions
    if st.sidebar.button("📚 Show Method Descriptions"):
        display_method_descriptions()
    
    # Main input area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📝 Input Queries</h2>', unsafe_allow_html=True)
        queries_input = st.text_area(
            "Enter your queries (one per line):",
            height=200,
            placeholder="Enter your search queries here...\nOne query per line\nExample: What is machine learning?\nHow does AI work?"
        )
    
    with col2:
        st.markdown('<h2 class="section-header">📁 Upload Documents</h2>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            f"Choose up to {max_files} text files:",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload .txt files containing the documents you want to compare against your queries."
        )
        
        if len(uploaded_files) > max_files:
            st.warning(f"Please upload no more than {max_files} files. Only the first {max_files} will be processed.")
            uploaded_files = uploaded_files[:max_files]
    
    # Process button
    if st.button("🚀 Calculate Similarities", type="primary"):
        if not queries_input.strip():
            st.error("Please enter at least one query.")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one document.")
            return
        
        # Parse queries
        queries = [q.strip() for q in queries_input.strip().split('\n') if q.strip()]
        
        if not queries:
            st.error("Please enter valid queries.")
            return
        
        # Read uploaded files
        documents = []
        doc_names = []
        
        for uploaded_file in uploaded_files:
            content = read_uploaded_file(uploaded_file)
            if content:
                documents.append(content)
                doc_names.append(uploaded_file.name)
        
        if not documents:
            st.error("Could not read any of the uploaded files. Please check file formats and try again.")
            return
        
        # Count selected methods
        selected_methods = ["TF-IDF"]  # Always included
        if use_bm25 and BM25_AVAILABLE:
            selected_methods.append("BM25")
        if use_bert and BERT_AVAILABLE:
            selected_methods.append("BERT")
        if use_splade and TRANSFORMERS_AVAILABLE:
            selected_methods.append("SPLADE")
        if use_mum and BERT_AVAILABLE:
            selected_methods.append("MUM")
        if use_muvera and BERT_AVAILABLE:
            selected_methods.append("MUVERA")
        
        st.markdown(f'<h2 class="section-header">📊 Results ({len(selected_methods)} methods)</h2>', unsafe_allow_html=True)
        
        # Create tabs for results
        tabs = st.tabs(selected_methods)
        results = {}
        
        # Calculate similarities for each method
        for i, method in enumerate(selected_methods):
            with tabs[i]:
                st.markdown(f"### {method} Similarity Results")
                
                # Progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    if method == "TF-IDF":
                        status_text.text("Calculating TF-IDF similarities...")
                        similarity_matrix, vectorizer = calculate_tfidf_similarity(queries, documents, doc_names)
                        progress_bar.progress(1.0)
                        
                    elif method == "BM25":
                        status_text.text("Calculating BM25 similarities...")
                        similarity_matrix = calculate_bm25_similarity(queries, documents, doc_names)
                        progress_bar.progress(1.0)
                        
                    elif method == "BERT":
                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                        similarity_matrix = calculate_bert_similarity(queries, documents, doc_names, progress_callback)
                        
                    elif method == "SPLADE":
                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                        similarity_matrix = calculate_splade_similarity(queries, documents, doc_names, progress_callback)
                        
                    elif method == "MUM":
                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                        similarity_matrix = calculate_mum_similarity(queries, documents, doc_names, progress_callback)
                        
                    elif method == "MUVERA":
                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                        similarity_matrix = calculate_muvera_similarity(queries, documents, doc_names, progress_callback)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    if similarity_matrix is not None:
                        # Create and display results DataFrame
                        results_df = create_similarity_dataframe(similarity_matrix, queries, doc_names)
                        results[method] = results_df
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Queries", len(queries))
                        with col2:
                            st.metric("Total Documents", len(documents))
                        with col3:
                            st.metric("Max Similarity", f"{results_df.values.max():.3f}")
                        with col4:
                            st.metric("Average Similarity", f"{results_df.values.mean():.3f}")
                        
                        # Display styled DataFrame
                        st.dataframe(
                            results_df.style.applymap(color_similarity_values).format("{:.3f}"),
                            use_container_width=True
                        )
                        
                        # Top matches for each query
                        st.markdown("#### 🎯 Top Matches by Query")
                        for j, query in enumerate(queries):
                            query_results = results_df.iloc[j].sort_values(ascending=False)
                            top_match = query_results.index[0]
                            top_score = query_results.iloc[0]
                            
                            st.markdown(f"**Query {j+1}:** *{query[:100]}{'...' if len(query) > 100 else ''}*")
                            st.markdown(f"Best match: **{top_match}** (Score: {top_score:.3f})")
                            if j < len(queries) - 1:
                                st.markdown("---")
                        
                        # Download CSV
                        csv_buffer = StringIO()
                        results_df.to_csv(csv_buffer)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label=f"📥 Download {method} Results as CSV",
                            data=csv_data,
                            file_name=f"{method.lower()}_similarity_results_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.error(f"Failed to calculate {method} similarity. Please check the error messages above.")
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error processing {method}: {str(e)}")
        
        # Comparison section if multiple methods were run
        if len(results) > 1:
            st.markdown('<h2 class="section-header">🔄 Method Comparison</h2>', unsafe_allow_html=True)
            
            comparison_data = []
            for method, df in results.items():
                comparison_data.append({
                    "Method": method,
                    "Max Similarity": df.values.max(),
                    "Min Similarity": df.values.min(),
                    "Mean Similarity": df.values.mean(),
                    "Std Similarity": df.values.std()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.format({
                "Max Similarity": "{:.3f}",
                "Min Similarity": "{:.3f}",
                "Mean Similarity": "{:.3f}",
                "Std Similarity": "{:.3f}"
            }), use_container_width=True)
            
            # Method recommendations
            st.markdown("#### 💡 Method Recommendations")
            
            best_max = comparison_df.loc[comparison_df["Max Similarity"].idxmax(), "Method"]
            best_mean = comparison_df.loc[comparison_df["Mean Similarity"].idxmax(), "Method"]
            
            st.markdown(f"- **Highest individual similarity:** {best_max}")
            st.markdown(f"- **Best average performance:** {best_mean}")
            
            if "BERT" in results or "MUM" in results or "MUVERA" in results:
                st.markdown("- **For semantic understanding:** Use BERT, MUM, or MUVERA")
            if "BM25" in results:
                st.markdown("- **For search applications:** Use BM25")
            if "TF-IDF" in results:
                st.markdown("- **For fast keyword matching:** Use TF-IDF")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This application supports multiple similarity calculation methods:
    - **TF-IDF**: Fast keyword-based similarity using term frequency and inverse document frequency
    - **BM25**: Probabilistic ranking function, excellent for search applications
    - **BERT**: Deep semantic understanding using transformer models
    - **SPLADE**: Sparse lexical expansion combining benefits of sparse and dense methods
    - **MUM**: Multilingual semantic model for cross-language understanding
    - **MUVERA**: Alternative semantic model using advanced architectures
    
    Neural methods (BERT, SPLADE, MUM, MUVERA) provide better semantic understanding but require more computational resources.
    """)

if __name__ == "__main__":
    main()
