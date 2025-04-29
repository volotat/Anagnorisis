# pages/text/engine.py

import os
import pickle
import hashlib
import datetime
import io
import time
import gc
import traceback # Import traceback

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer # Using SentenceTransformer for easier loading/encoding
from huggingface_hub import snapshot_download

# Assuming file_manager and db_models are structured correctly
import src.scoring_models
import pages.file_manager as file_manager
import pages.text.db_models as db_models # Import DB model for potential future use/reference


from scipy.spatial.distance import cosine

# Placeholder for fast cache 
files_embeds_fast_cache = {}

def get_text_metadata(file_path):
    """
    Extracts basic metadata from a text file.
    Can be extended later for more fields like creation date, modification date, etc.
    """
    metadata = {}
    try:
        # Get file size
        metadata['file_size'] = os.path.getsize(file_path)
        # Get creation time (Unix timestamp)
        metadata['creation_time'] = os.path.getctime(file_path)
        # Get modification time (Unix timestamp)
        metadata['modification_time'] = os.path.getmtime(file_path)
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        metadata['file_size'] = None
        metadata['creation_time'] = None
        metadata['modification_time'] = None

    return metadata

class TextSearch:
    device = None
    model = None
    tokenizer = None # SentenceTransformer model includes a tokenizer
    is_busy = False # Simple flag to prevent concurrent processing (needs refinement for production)

    model_hash = None
    embedding_dim = None

    cached_file_list = None
    cached_file_hash = None
    cached_metadata = None

    @staticmethod
    def initiate(cfg, models_folder='./models', cache_folder='./cache'):
        """
        Initializes the TextSearch engine by loading the SentenceTransformer model
        and setting up necessary components.
        """
        if TextSearch.model is not None:
            # Already initialized
            return

        model_name_id = cfg.text.embedding_model # e.g., "jinaai/jina-embeddings-v3"
        # Create a safe folder name from the model ID
        model_folder_name = model_name_id.replace('/', '__') # Replace slashes
        local_model_path = os.path.join(models_folder, model_folder_name)

        # Ensure base models directory exists
        os.makedirs(models_folder, exist_ok=True)

        # Check if model exists locally, if not, download it
        # Check for a common file like 'config.json' to indicate a potentially complete download
        config_path = os.path.join(local_model_path, 'config.json')
        if not os.path.exists(config_path):
            print(f"Model '{model_name_id}' not found locally or incomplete at '{local_model_path}'. Downloading...")
            try:
                snapshot_download(
                    repo_id=model_name_id,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False # Download actual files
                )
                print(f"Model '{model_name_id}' downloaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to download model '{model_name_id}'. Please check your internet connection and permissions.")
                print(f"Error details: {e}")
                raise RuntimeError(f"Failed to download required model: {model_name_id}") from e
        else:
            print(f"Found existing model '{model_name_id}' at '{local_model_path}'.")

        # Now load the model from the guaranteed local path using SentenceTransformer
        try:
            TextSearch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {TextSearch.device}")
            # Use SentenceTransformer.load to handle model + tokenizer + pooling correctly
            # Pass trust_remote_code=True as required by the Jina model
            TextSearch.model = SentenceTransformer(local_model_path, device=TextSearch.device, trust_remote_code=True)
            # SentenceTransformer loads its own tokenizer internally
            TextSearch.tokenizer = TextSearch.model.tokenizer

            # Get embedding dimension after loading
            TextSearch.embedding_dim = TextSearch.model.get_sentence_embedding_dimension()
            print(f"Text embedding dimension: {TextSearch.embedding_dim}")

            # Calculate model hash based on primary weights file
            TextSearch.model_hash = TextSearch.get_model_hash(local_model_path)
            print(f"Model hash: {TextSearch.model_hash}")

        except Exception as e:
            print(f"ERROR: Failed to load model '{model_name_id}' from '{local_model_path}'. The download might be incomplete or corrupted.")
            print(f"Error details: {e}")
            raise RuntimeError(f"Failed to load required model: {model_name_id}") from e

        # Caches are initialized externally (e.g., in app.py or serve.py)
        # We could pass them here or access them via the file_manager module if needed later.
        print("TextSearch initiated successfully.")

        # --- Initialize Caches ---
        # Ensure cache directory exists
        os.makedirs(cache_folder, exist_ok=True)
        TextSearch.cached_file_list = file_manager.CachedFileList(os.path.join(cache_folder, 'text_file_list.pkl'))
        TextSearch.cached_file_hash = file_manager.CachedFileHash(os.path.join(cache_folder, 'text_file_hash.pkl'))
        # TextSearch.cached_metadata = file_manager.CachedMetadata(os.path.join(cache_folder, 'text_metadata.pkl'), get_text_metadata)


    @staticmethod
    def get_model_hash(model_path):
        """
        Calculates a hash of the primary weights file of the SentenceTransformer model.
        This hash serves as the embedder version.
        """
        weights_files = ['pytorch_model.bin', 'model.safetensors'] # Common weight file names
        primary_weights_file = None

        for wf in weights_files:
            full_path = os.path.join(model_path, wf)
            if os.path.exists(full_path):
                primary_weights_file = full_path
                break

        if not primary_weights_file:
            # Fallback: Hash a stable config file if weights aren't found (less ideal)
            config_file = os.path.join(model_path, 'config.json')
            if os.path.exists(config_file):
                print(f"Warning: No standard weights file found in {model_path}. Hashing config.json instead.")
                primary_weights_file = config_file
            else:
                print(f"Error: Cannot find weights or config.json in {model_path} to calculate hash.")
                return "hash_error_no_file"

        try:
            with open(primary_weights_file, "rb") as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192): # Read 8KB chunks
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {primary_weights_file}: {e}")
            return "hash_error_read_failed"


    @staticmethod
    def process_text(query_text, cfg):
        """
        Generates an embedding for a search query string.
        Uses 'retrieval.query' task and configured truncation dimension.
        """
        if TextSearch.model is None:
            print("Error: TextSearch model not initialized.")
            return None

        # Basic busy check (can be improved)
        if TextSearch.is_busy:
            print("Warning: TextSearch engine is busy. Skipping query embedding.")
            return None

        TextSearch.is_busy = True
        try:
            print(f"Embedding query: '{query_text[:50]}...'")
            # SentenceTransformer handles pooling and normalization based on the model's config.
            # Pass task and truncate_dim directly to encode.
            query_embedding = TextSearch.model.encode(
                query_text,
                task="retrieval.query", # Task for query embedding
                truncate_dim=cfg.text.embedding_dimension,
                convert_to_tensor=True, # Return as torch tensor
                device=TextSearch.device
            )
            print(f"Query embedding shape: {query_embedding.shape}")
            return query_embedding # Return as a PyTorch tensor [1, embedding_dim]
        except Exception as e:
            print(f"Error processing text query '{query_text}': {e}")
            traceback.print_exc()
            # Return a tensor of zeros with the correct shape on error
            dim = cfg.text.embedding_dimension or TextSearch.embedding_dim
            return torch.zeros((1, dim), device=TextSearch.device) if dim else None
        finally:
            TextSearch.is_busy = False


    @staticmethod
    def process_file(file_path, cfg, callback=None):
        """
        Reads a text file, splits it into chunks, and generates embeddings for each chunk.
        Uses 'retrieval.passage' task and configured chunking/truncation.
        Does NOT interact with DB in this phase. Returns list of numpy arrays.
        """
        if TextSearch.model is None:
            print("Error: TextSearch model not initialized. Cannot process file.")
            return []

        if TextSearch.is_busy:
            print(f"Warning: TextSearch engine is busy. Skipping file: {os.path.basename(file_path)}")
            return []

        TextSearch.is_busy = True
        print(f"Processing file: {file_path}")

        try:
            # Check if the file is already in the fast cache
            file_hash = TextSearch.cached_file_hash.get_file_hash(file_path)
            
            if file_hash in files_embeds_fast_cache:
                return files_embeds_fast_cache[file_hash]

            # 1. Read File Content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading file content {file_path}: {e}")
                traceback.print_exc()
                return []

            # 2. Tokenize and Chunk
            # Use the model's tokenizer
            # We tokenize the whole document first to handle chunking based on token count
            print(f"Tokenizing {len(content)} characters...")
            tokens = TextSearch.tokenizer(content, add_special_tokens=False, truncation=False)["input_ids"]
            file_token_length = len(tokens)
            print(f"Total tokens: {file_token_length}")

            chunk_size = cfg.text.chunk_size
            chunk_overlap = cfg.text.chunk_overlap

            chunk_texts = []
            start_token_idx = 0
            while start_token_idx < file_token_length:
                end_token_idx = min(start_token_idx + chunk_size, file_token_length)
                # Decode the token slice for this chunk
                chunk_text = TextSearch.tokenizer.decode(tokens[start_token_idx:end_token_idx])
                chunk_texts.append(chunk_text)

                # Calculate next chunk start
                next_start_token_idx = start_token_idx + chunk_size - chunk_overlap
                # If overlap is too large or we're near the end, simply advance by chunk size
                if next_start_token_idx <= start_token_idx:
                    next_start_token_idx = start_token_idx + chunk_size

                # Break if the last chunk was added
                if end_token_idx == file_token_length:
                    break

                start_token_idx = next_start_token_idx

            print(f"Split into {len(chunk_texts)} chunks.")
            if not chunk_texts:
                 print("No chunks generated (empty file?).")
                 return []


            # 3. Generate Embeddings for Chunks
            print(f"Embedding {len(chunk_texts)} chunks...")
            #if callback:
            #     # Initial progress update after chunking
            #     callback(0, len(chunk_texts), f"Embedding {len(chunk_texts)} chunks for {os.path.basename(file_path)}...")

            # Process in batches using SentenceTransformer's encode method
            embedding_batch_size = 32 # Adjust based on VRAM
            chunk_embeddings_list_np = []

            for i in tqdm(range(0, len(chunk_texts), embedding_batch_size), desc=f"Embedding {os.path.basename(file_path)}"):
                batch_texts = chunk_texts[i:i+embedding_batch_size]
                try:
                    # Encode the batch
                    batch_embeddings_np = TextSearch.model.encode(
                        batch_texts,
                        task="retrieval.passage", # Task for document passages
                        truncate_dim=cfg.text.embedding_dimension,
                        convert_to_tensor=False, # Get numpy arrays directly
                        device=TextSearch.device,
                        batch_size=embedding_batch_size # Pass batch size to encode
                    )
                    chunk_embeddings_list_np.extend(batch_embeddings_np)
                except Exception as e:
                    print(f"Error encoding batch {i//embedding_batch_size} for {file_path}: {e}")
                    traceback.print_exc()
                    # Append zero vectors or handle error as needed
                    error_batch_size = len(batch_texts)
                    dim = cfg.text.embedding_dimension or TextSearch.embedding_dim
                    if dim:
                        zero_embeddings = np.zeros((error_batch_size, dim), dtype=np.float32)
                        chunk_embeddings_list_np.extend(zero_embeddings)

                #if callback:
                #    processed_chunks = min(i + embedding_batch_size, len(chunk_texts))
                #    # Callback reports progress (e.g., to update UI status)
                #    callback(processed_chunks, len(chunk_texts), f"Embedding chunks for {os.path.basename(file_path)}")

            print(f"Generated {len(chunk_embeddings_list_np)} embeddings.")
            
            # Store in fast cache for future use
            files_embeds_fast_cache[file_hash] = chunk_embeddings_list_np

            return chunk_embeddings_list_np # Return list of numpy arrays

        except Exception as e:
            print(f"Critical error processing file {file_path}: {e}")
            traceback.print_exc()
            return [] # Return empty list on critical error
        finally:
            TextSearch.is_busy = False # Ensure busy flag is always reset

    @staticmethod
    def process_files(file_paths, cfg, callback=None):
        """
        Processes multiple files and generates embeddings for each.
        Returns a list of tuples (file_path, chunk_embeddings).
        """
        all_results = []
        for ind, file_path in enumerate(file_paths):
            chunk_embeddings = TextSearch.process_file(file_path, cfg, callback)
            all_results.append(chunk_embeddings)

            if callback:
                # Initial progress update after chunking
                callback(ind, len(file_paths))

        #print([result for result in all_results])
        # Concatenate all embeddings
        #all_embeds = torch.cat(all_results, dim=0)

        return all_results
    
    @staticmethod
    def compare(file_embeddings, query_embedding):
        """
        Compares a query embedding against file embeddings to calculate relevance scores.
        
        Args:
            file_embeddings: A list where each item contains chunk embeddings for a file
                            (List[List[numpy.ndarray]])
            query_embedding: The embedding of the search query (torch.Tensor)
        
        Returns:
            numpy.ndarray: An array of scores, one per file, representing the maximum 
                        similarity across all chunks in each file
        """
        # Ensure query embedding is numpy array with float32 dtype
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.to(torch.float32).cpu().numpy()
        
        # Calculate max similarity score for each file
        scores = []
        for file_chunks in file_embeddings:
            # Handle empty chunk lists
            if not file_chunks or len(file_chunks) == 0:
                scores.append(0.0)
                continue
                
            # Calculate similarity for each chunk
            chunk_scores = []
            for chunk_embedding in file_chunks:
                # Convert to numpy if needed and ensure float32 dtype
                if isinstance(chunk_embedding, torch.Tensor):
                    chunk_embedding = chunk_embedding.to(torch.float32).cpu().numpy()
                else:
                    # Ensure numpy arrays are also float32
                    chunk_embedding = chunk_embedding.astype(np.float32)
                    
                print(f"Chunk embedding shape: {chunk_embedding.shape}")
                print(f"Query embedding shape: {query_embedding.shape}")
                
                # Calculate cosine similarity: 1 - cosine distance
                # Higher value means more similar (closer to 1)
                similarity = 1 - cosine(query_embedding, chunk_embedding)
                chunk_scores.append(similarity)
                
            # Take maximum similarity across all chunks as the file's score
            # This represents the most relevant chunk in the file
            scores.append(max(chunk_scores) if chunk_scores else 0.0)
            
        return np.array(scores)
    
# Create scoring model singleton class
class TextEvaluator(src.scoring_models.Evaluator):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_dim=1024, rate_classes=11):
        if not hasattr(self, '_initialized'):
            super(TextEvaluator, self).__init__(embedding_dim, rate_classes)
            self._initialized = True

# --- Testing Section ---
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import tempfile
    import glob
    import re
    from scipy.spatial.distance import cosine
    import colorama  # For colored terminal output

    # Initialize colorama for cross-platform colored terminal output
    colorama.init()

    print("--- Running TextSearch Engine Test ---")

    # Create a dummy config for testing
    dummy_cfg_dict = {
        'main': {
            'models_path': './models',
            'cache_path': './cache'
        },
        'text': {
            'embedding_model': "jinaai/jina-embeddings-v3",
            'chunk_size': 512,  # Larger chunk size for realistic content
            'chunk_overlap': 100,
            'embedding_dimension': 0  # Use model's native dimension
        }
    }
    cfg = OmegaConf.create(dummy_cfg_dict)

    # Ensure directories exist
    os.makedirs(cfg.main.models_path, exist_ok=True)
    os.makedirs(cfg.main.cache_path, exist_ok=True)

    # --- Initialize the model ---
    try:
        print("\nInitializing TextSearch engine...")
        TextSearch.initiate(cfg, cfg.main.models_path, cfg.main.cache_path)
        print(f"Model loaded successfully. Embedding dimension: {TextSearch.embedding_dim}")
    except Exception as e:
        print(f"FATAL: Model initiation failed: {e}")
        traceback.print_exc()
        exit(1)

    # --- Load the test files from the provided directory ---
    test_dir = os.path.join(os.path.dirname(__file__), "engine_test_data")
    if not os.path.exists(test_dir):
        print(f"Error: Test data directory not found at {test_dir}")
        exit(1)
        
    test_files = glob.glob(os.path.join(test_dir, "*.txt"))
    if not test_files:
        print(f"Error: No .txt files found in {test_dir}")
        exit(1)
        
    print(f"\nFound {len(test_files)} test files in {test_dir}:")
    for file_path in test_files:
        print(f"  - {os.path.basename(file_path)}")

    # --- Set up Search Functionality with Caching ---
    # Cache for file embeddings to avoid reprocessing
    file_embeddings_cache = {}
    file_chunks_cache = {}
    
    def process_and_cache_file(file_path):
        """Process a file and cache its embeddings and chunks"""
        if file_path in file_embeddings_cache:
            return file_embeddings_cache[file_path], file_chunks_cache[file_path]
            
        print(f"Processing {os.path.basename(file_path)} (not in cache)...")
        
        # Generate embeddings for the file
        chunk_embeddings = TextSearch.process_file(file_path, cfg)
        
        if not chunk_embeddings:
            print(f"No embeddings generated for {file_path}")
            return [], []
            
        # Read file content to get chunks
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split into chunks to match embeddings
        tokens = TextSearch.tokenizer(content, add_special_tokens=False, truncation=False)["input_ids"]
        chunk_size = cfg.text.chunk_size
        chunk_overlap = cfg.text.chunk_overlap
        
        # Generate chunks
        chunk_texts = []
        start_token_idx = 0
        while start_token_idx < len(tokens):
            end_token_idx = min(start_token_idx + chunk_size, len(tokens))
            chunk_text = TextSearch.tokenizer.decode(tokens[start_token_idx:end_token_idx])
            chunk_texts.append(chunk_text)
            
            # Calculate next position
            next_start_token_idx = start_token_idx + chunk_size - chunk_overlap
            if next_start_token_idx <= start_token_idx:
                next_start_token_idx = start_token_idx + chunk_size
                
            if end_token_idx == len(tokens):
                break
                
            start_token_idx = next_start_token_idx
            
        # Cache the results
        file_embeddings_cache[file_path] = chunk_embeddings
        file_chunks_cache[file_path] = chunk_texts
        
        return chunk_embeddings, chunk_texts

    def calculate_similarity(embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        # Convert tensors to numpy if needed
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.cpu().numpy()
            
        # Ensure embeddings are flattened
        if len(embedding1.shape) > 1:
            embedding1 = embedding1.flatten()
        if len(embedding2.shape) > 1:
            embedding2 = embedding2.flatten()
            
        # Calculate cosine similarity (1 - cosine distance)
        return 1 - cosine(embedding1, embedding2)

    def highlight_text(text, keywords, context_size=50):
        """Highlight keywords in text and provide context"""
        # Prepare text for display with highlighting
        highlighted = text
        for keyword in keywords:
            if len(keyword) < 3:  # Skip very short words
                continue
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f"{colorama.Fore.GREEN}{keyword}{colorama.Style.RESET_ALL}", highlighted)
        
        # Find a good match to center the context window
        match_pos = None
        for keyword in keywords:
            if len(keyword) < 3:
                continue
            match = re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)
            if match:
                match_pos = match.start()
                break
                
        if match_pos is not None:
            start = max(0, match_pos - context_size)
            end = min(len(text), match_pos + context_size)
            
            # Add ellipsis indicators
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            
            return prefix + highlighted[start:end] + suffix
        
        # If no match found, return beginning of text
        return highlighted[:100] + "..."

    # Cache for query embeddings
    query_embeddings_cache = {}
    
    def semantic_search(query, test_files, top_k=3):
        """Perform semantic search across the test files with caching"""
        print(f"\n{colorama.Fore.CYAN}Searching for: '{query}'{colorama.Style.RESET_ALL}")
        
        # Process query and cache embedding
        query_keywords = [w.lower() for w in query.split() if len(w) > 2]
        
        # Check if query embedding is already cached
        if query in query_embeddings_cache:
            print("Using cached query embedding")
            query_embedding = query_embeddings_cache[query]
        else:
            print("Generating new query embedding")
            query_embedding = TextSearch.process_text(query, cfg)
            query_embeddings_cache[query] = query_embedding
        
        # Store results
        all_results = []
        
        # Process each file (using cached embeddings when available)
        for file_path in test_files:
            chunk_embeddings, chunk_texts = process_and_cache_file(file_path)
            
            if not chunk_embeddings:
                continue
                
            # Calculate similarity for each chunk
            for i, (chunk_text, chunk_embedding) in enumerate(zip(chunk_texts, chunk_embeddings)):
                similarity = calculate_similarity(query_embedding, chunk_embedding)
                
                # Store result
                all_results.append({
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'chunk_idx': i,
                    'text': chunk_text,
                    'similarity': similarity
                })
                
        # Sort by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        return all_results[:top_k]

    # --- Pre-process and cache all files at startup ---
    print("\nPre-processing all files to build cache...")
    for file_path in test_files:
        _ = process_and_cache_file(file_path)
    print("All files processed and cached.")

    # --- Execute Realistic Search Queries ---
    print("\n" + "="*80)
    print(f"{colorama.Fore.YELLOW}REALISTIC SEMANTIC SEARCH TEST{colorama.Style.RESET_ALL}")
    print("="*80)
    
    # Define realistic search queries
    search_queries = [
        "How do quantum computers use superposition?",
        "What is the difference between trapped ions and superconducting qubits?",
        "How to create visualizations with pandas and matplotlib",
        "Steps to handle missing values in data analysis",
        "Recipe for authentic Italian carbonara",
        "What temperature should I bake pizza at?"
    ]
    
    for query in search_queries:
        # Perform search
        results = semantic_search(query, test_files)
        
        # Display results
        if results:
            print(f"\n{colorama.Fore.BLUE}Found {len(results)} matches for '{query}'{colorama.Style.RESET_ALL}")
            
            for i, result in enumerate(results):
                file_name = result['file_name']
                similarity = result['similarity'] * 100  # Convert to percentage
                
                print(f"\n{colorama.Fore.YELLOW}Match #{i+1}{colorama.Style.RESET_ALL}: {file_name} ({similarity:.1f}% match)")
                
                # Show highlighted excerpt
                highlighted = highlight_text(
                    result['text'],
                    query.lower().split(),
                    context_size=200  # Larger context for better readability
                )
                print(f"{colorama.Fore.WHITE}{highlighted}{colorama.Style.RESET_ALL}")
                print("-" * 80)
        else:
            print(f"No matches found for '{query}'")
    
    print("\n--- TextSearch Engine Test Completed ---")