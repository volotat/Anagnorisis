import os
import torch
import numpy as np
import traceback
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from scipy.spatial.distance import cosine

from src.model_manager import ModelManager

class TextEmbedder:
    """
    A singleton class to handle text chunking, embedding, and comparison.
    It manages a single sentence-transformer model instance using ModelManager
    for efficient GPU memory usage.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cfg=None):
        if self._initialized:
            return
        
        if cfg is None:
            raise ValueError("TextEmbedder requires a configuration object (cfg) on first initialization.")
            
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        self._initialized = True

    def initiate(self, models_folder: str):
        """
        Initializes the embedder by downloading and loading the model.
        This method is safe to call multiple times.
        """
        if self.model is not None:
            return # Already initiated

        model_name = self.cfg.text.embedding_model
        if not model_name:
            raise ValueError("cfg.text.embedding_model is not specified.")

        model_folder_name = model_name.replace('/', '__')
        local_model_path = os.path.join(models_folder, model_folder_name)

        self._ensure_model_downloaded(models_folder, model_name)
        self._load_model_and_tokenizer(local_model_path)

    def _ensure_model_downloaded(self, models_folder: str, model_name: str):
        # Define paths for both the main model and its dependency
        main_model_path = os.path.join(models_folder, model_name.replace('/', '__'))
        # dependency_repo_id = "jinaai/xlm-roberta-flash-implementation"
        # dependency_model_path = os.path.join(models_folder, dependency_repo_id.replace('/', '__'))

        # 1. Check and download the main model
        if not os.path.exists(os.path.join(main_model_path, 'config.json')):
            print(f"TextEmbedder: Main model '{model_name}' not found locally. Downloading...")
            try:
                snapshot_download(
                    repo_id=model_name,
                    local_dir=main_model_path,
                    local_dir_use_symlinks=False,
                    # revision="f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"  # Specific commit for consistency
                )
                print(f"TextEmbedder: Main model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download main model '{model_name}': {e}") from e
        else:
            print(f"TextEmbedder: Found existing main model '{model_name}'.")

        # # 2. Check and download the dependency model
        # if not os.path.exists(os.path.join(dependency_model_path, 'config.json')):
        #     print(f"TextEmbedder: Dependency '{dependency_repo_id}' not found locally. Downloading...")
        #     try:
        #         snapshot_download(
        #             repo_id=dependency_repo_id,
        #             #local_dir=dependency_model_path,
        #             local_dir_use_symlinks=False,
        #             revision="9dc60336f6b2df56c4f094dd287ca49fb7b93342"
        #         )
        #         print(f"TextEmbedder: Dependency downloaded successfully.")
        #     except Exception as e:
        #         raise RuntimeError(f"Failed to download dependency '{dependency_repo_id}': {e}") from e
        # else:
        #     print(f"TextEmbedder: Found existing dependency '{dependency_repo_id}'.")

    def _load_model_and_tokenizer(self, local_path: str):
        try:
            print("TextEmbedder: Loading model to CPU before wrapping with ModelManager...")
            # Load model to CPU first
            raw_model = SentenceTransformer(
                local_path, 
                device='cpu', 
                trust_remote_code=True,
                # local_files_only=True,
                # revision="f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9", 
            )
            self.tokenizer = raw_model.tokenizer
            self.embedding_dim = self.cfg.text.embedding_dimension or raw_model.get_sentence_embedding_dimension()
            
            # Wrap the raw model with ModelManager
            self.model = ModelManager(raw_model, device=self.device)
            print(f"TextEmbedder initiated. Embedding dim: {self.embedding_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to load text embedding model from '{local_path}': {e}") from e

    def get_chunk_scores(self, long_text: str, query_text: str) -> dict[str, float]:
        """
        Takes a long text and a query, breaks the text into segments (handling overlaps),
        and returns a dictionary mapping each text segment to its relevance score.
        Overlapping segments have their scores averaged from the parent chunks.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("TextEmbedder not initiated. Call initiate() first.")

        # 1. Embed the query and the text chunks
        query_embedding = self.embed_query(query_text)
        all_chunk_embeddings = self.embed_text(long_text)

        if all_chunk_embeddings is None or len(all_chunk_embeddings) == 0:
            return {}
        
        if query_embedding is None or len(query_embedding) == 0:
            return {}

        # 2. Re-create chunk character offsets for segmentation logic.
        # This is necessary because embed_text only returns embeddings.
        encoding = self.tokenizer(
            long_text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
        )
        tokens = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        
        chunk_size = self.cfg.text.chunk_size
        chunk_overlap = self.cfg.text.chunk_overlap
        
        chunks_info = []
        start_token_idx = 0
        
        while start_token_idx < len(tokens):
            end_token_idx = min(start_token_idx + chunk_size, len(tokens))
            char_start = offsets[start_token_idx][0]
            char_end = offsets[end_token_idx - 1][1]
            
            chunks_info.append({'start_char': char_start, 'end_char': char_end})
            
            next_start_token_idx = start_token_idx + chunk_size - chunk_overlap
            if next_start_token_idx <= start_token_idx:
                next_start_token_idx = start_token_idx + chunk_size
            if end_token_idx == len(tokens):
                break
            start_token_idx = next_start_token_idx

        # 3. Calculate scores for each chunk using the compare method
        all_chunk_scores = self.compare(all_chunk_embeddings, query_embedding)
        for i, score in enumerate(all_chunk_scores):
            chunks_info[i]['score'] = score

        # 4. Process segments and calculate final scores using character indices
        segments = {}
        last_processed_char_index = 0

        for i in range(len(chunks_info)):
            current_chunk = chunks_info[i]
            
            if i + 1 < len(chunks_info):
                next_chunk = chunks_info[i+1]
                unique_part_end_char = next_chunk['start_char']
                
                if last_processed_char_index < unique_part_end_char:
                    unique_text = long_text[last_processed_char_index:unique_part_end_char]
                    if unique_text:
                        segments[unique_text] = current_chunk['score']
                    last_processed_char_index = unique_part_end_char

                overlap_start_char = next_chunk['start_char']
                overlap_end_char = current_chunk['end_char']

                if overlap_start_char < overlap_end_char:
                    overlap_text = long_text[overlap_start_char:overlap_end_char]
                    if overlap_text:
                        averaged_score = (current_chunk['score'] + next_chunk['score']) / 2.0
                        segments[overlap_text] = averaged_score
                    last_processed_char_index = overlap_end_char
            else:
                final_segment_start_char = last_processed_char_index
                final_segment_end_char = current_chunk['end_char']
                if final_segment_start_char < final_segment_end_char:
                    final_text = long_text[final_segment_start_char:final_segment_end_char]
                    if final_text:
                        segments[final_text] = current_chunk['score']
        
        return segments

    def embed_text(self, long_text: str) -> np.ndarray:
        """
        Takes a long string of text, splits it into chunks using precise token offsets,
        and returns a list of embeddings for these chunks.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("TextEmbedder not initiated. Call initiate() first.")

        try:
            # Chunk the long text using offset mapping for perfect reconstruction
            encoding = self.tokenizer(
                long_text, 
                add_special_tokens=False, 
                truncation=False, 
                return_offsets_mapping=True
            )
            tokens = encoding["input_ids"]
            offsets = encoding["offset_mapping"]
            
            chunk_size = self.cfg.text.chunk_size
            chunk_overlap = self.cfg.text.chunk_overlap
            
            chunk_texts = []
            start_token_idx = 0
            
            while start_token_idx < len(tokens):
                end_token_idx = min(start_token_idx + chunk_size, len(tokens))
                
                char_start = offsets[start_token_idx][0]
                char_end = offsets[end_token_idx - 1][1]
                
                chunk_text = long_text[char_start:char_end]
                chunk_texts.append(chunk_text)
                
                next_start_token_idx = start_token_idx + chunk_size - chunk_overlap
                if next_start_token_idx <= start_token_idx:
                    next_start_token_idx = start_token_idx + chunk_size
                if end_token_idx == len(tokens):
                    break
                start_token_idx = next_start_token_idx

            if not chunk_texts:
                return []

            # Use the managed model instance to get embeddings
            embeddings = self.model.encode(
                chunk_texts,
                task="retrieval.passage",
                truncate_dim=self.cfg.text.embedding_dimension,
                convert_to_tensor=False,
                device=self.device
            )

            return embeddings

        except Exception as e:
            print(f"Error embedding long text: {e}")
            traceback.print_exc()
            return []

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Takes a short query string and returns a single embedding as a numpy array.
        """
        if not self.model:
            raise RuntimeError("TextEmbedder not initiated. Call initiate() first.")
        
        try:
            embedding = self.model.encode(
                query_text,
                task="retrieval.query",
                truncate_dim=self.cfg.text.embedding_dimension,
                convert_to_tensor=False,
                device=self.device
            )
            return embedding
        except Exception as e:
            print(f"Error embedding query '{query_text}': {e}")
            traceback.print_exc()
            dim = self.cfg.text.embedding_dimension or self.embedding_dim
            return np.zeros((dim,), dtype=np.float32) if dim else np.array([])

    def compare(self, text_embeddings: np.ndarray, query_embedding: np.ndarray) -> list[float]:
        """
        Compares a query embedding against an array of text chunk embeddings using
        a vectorized dot product for fast and faithful similarity calculation.
        """
        if text_embeddings is None or query_embedding is None or text_embeddings.size == 0 or query_embedding.size == 0:
            return [0.0]

        # Ensure embeddings are normalized (good practice, though often pre-normalized)
        # Jina embeddings are already normalized, so this is a safeguard.
        norm_query = query_embedding  / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        norm_chunks = text_embeddings  / np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        # Perform vectorized dot product
        # (num_chunks, dim) @ (dim, 1) -> (num_chunks,)
        scores = np.dot(norm_chunks, norm_query.T).flatten()
        return scores.tolist()
    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    import pprint

    # 1. Setup mock configuration
    mock_cfg = OmegaConf.create({
        'text': {
            'embedding_model': 'jinaai__jina-embeddings-v3', #jinaai__jina-embeddings-v3
            'embedding_dimension': 512,
            'chunk_size': 192,
            'chunk_overlap': 64,
        }
    })

    # 2. Create a dummy text file for testing
    script_dir = os.path.dirname(__file__)
    test_text = (
        "Section 1: A Brief History of the Roman Empire.\n"
        "The Roman Empire, which succeeded the Roman Republic, was characterized by a government headed by emperors and large territorial holdings around the Mediterranean Sea in Europe, Africa, and Asia. "
        "The first emperor, Augustus, ruled from 27 BC to 14 AD, marking the beginning of the Pax Romana, a period of relative peace and stability that lasted for over two centuries. "
        "The empire's economy was based on agriculture and trade, with a vast network of roads and sea routes facilitating the movement of goods and armies. "
        "Key challenges included managing its vast borders, internal political instability, and economic pressures. "
        "The eventual decline in the West during the 5th century AD was a complex process attributed to factors like barbarian invasions, economic troubles, and over-reliance on slave labor.\n\n"
        
        "Section 2: Fundamentals of Quantum Computing.\n"
        "Quantum computing represents a paradigm shift from classical computing. Instead of bits, which can be a 0 or a 1, quantum computers use qubits. "
        "A qubit can exist in a state of superposition, meaning it can be a combination of both 0 and 1 simultaneously. This property allows quantum computers to process a vast number of calculations at once. "
        "Another key principle is entanglement, a phenomenon where two or more qubits become linked in such a way that their fates are intertwined, regardless of the distance separating them. "
        "Measuring the state of one entangled qubit instantly influences the state of the other. These principles could enable quantum machines to solve complex optimization, cryptography, and simulation problems that are intractable for even the most powerful classical supercomputers.\n\n"

        "Section 3: The Art of Baking Sourdough Bread.\n"
        "Baking sourdough bread is a rewarding process that relies on a symbiotic culture of wild yeast and lactobacilli, known as a sourdough starter. "
        "The process begins with feeding the starter with flour and water until it becomes active and bubbly. The dough is then mixed, often using a technique called autolyse where flour and water are combined before adding the starter and salt. "
        "This is followed by a period of bulk fermentation, which includes a series of 'stretch and folds' to develop gluten strength. "
        "After shaping the loaf, it undergoes a final proof, often in a cool environment, before being baked in a very hot oven, typically inside a Dutch oven to trap steam and create a crispy crust."
    )

    # 4. Define a query and run the test
    query = "What are the principles of quantum superposition and entanglement?"
    
    # 3. Initialize the TextEmbedder
    # Note: The models will be downloaded to ../models/ relative to this script
    models_path = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    os.makedirs(models_path, exist_ok=True)
    
    embedder = TextEmbedder(cfg=mock_cfg)
    embedder.initiate(models_folder=models_path)

    # 3.5. Pre-run type and shape checks
    print("\n" + "="*50)
    print("Running pre-flight checks on method outputs...")
    
    # Test embed_query
    test_query_embedding = embedder.embed_query(query)
    assert isinstance(test_query_embedding, np.ndarray), f"embed_query should return np.ndarray, but got {type(test_query_embedding)}"
    assert test_query_embedding.shape == (mock_cfg.text.embedding_dimension,), f"embed_query returned wrong shape: {test_query_embedding.shape}"
    print("✅ embed_query returns correct type and shape.")

    # Test embed_text
    test_text_embeddings = embedder.embed_text(test_text)

    # The result can be a single ndarray (for one chunk) or a list of them.
    if isinstance(test_text_embeddings, np.ndarray):
        # This case handles single-chunk inputs
        assert test_text_embeddings[0].shape == (mock_cfg.text.embedding_dimension,), f"embed_text (single chunk) returned wrong shape: {test_text_embeddings[0].shape}"
    else:
        raise AssertionError(f"embed_text returned an unexpected type: {type(test_text_embeddings)}")
    print("✅ embed_text returns correct type and shape.")

    # Test compare
    test_scores = embedder.compare(test_text_embeddings, test_query_embedding)
    if isinstance(test_scores, list):
        assert all(isinstance(score, float) for score in test_scores), f"compare should return a list of floats, but got {type(test_scores)}"
    else:
        raise AssertionError(f"compare returned an unexpected type: {type(test_scores)}")
    print("✅ compare returns correct type.")

    print("Pre-flight checks passed.")
    print("="*50 + "\n")


    print("\n" + "="*50)
    print(f"Query: '{query}'")
    print("="*50 + "\n")
    
    # Get the scored chunks
    scored_segments = embedder.get_chunk_scores(test_text, query)

    # 5. Print the results
    print("Scored Text Segments:")
    # Use a custom print for better readability
    for text, score in scored_segments.items():
        print(f"  - SCORE: {score:.4f} | TEXT: \"{text.strip()}\"")

    # 6. Verify that the combined segments reconstruct the original text
    reconstructed_text = "".join(scored_segments.keys())
    
    print("\n" + "="*50)
    print("Verifying text reconstruction...")
    
    if reconstructed_text == test_text:
        print("✅ SUCCESS: Reconstructed text perfectly matches the original text.")
    else:
        print("❌ FAILURE: Reconstructed text does not match the original text.")
        # For debugging, print the differences
        import difflib
        diff = difflib.unified_diff(
            test_text.splitlines(keepends=True),
            reconstructed_text.splitlines(keepends=True),
            fromfile='original_text',
            tofile='reconstructed_text',
        )
        print("Differences found:\n" + ''.join(diff))