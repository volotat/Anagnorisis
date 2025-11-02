import os
import torch
import numpy as np
import gc

from pages.utils import SortingProgressCallback, EmbeddingGatheringCallback

from src.metadata_search import MetadataSearch

import rapidfuzz 
import unicodedata
import re

def compute_distances_batched(embeds_media, batch_size=1024 * 24):
    """Computes pairwise distances in batches to conserve GPU memory."""
    embeds_media = torch.tensor(embeds_media, dtype=torch.float32)
    num_items = embeds_media.shape[0]
    distances = torch.zeros((num_items, num_items), dtype=torch.float32)
    
    for start_row in range(0, num_items, batch_size):
        end_row = min(start_row + batch_size, num_items)
        batch = embeds_media[start_row:end_row].cuda()
        
        for start_col in range(0, num_items, batch_size):
            end_col = min(start_col + batch_size, num_items)
            compare_batch = embeds_media[start_col:end_col].cuda()
            dists_batch = torch.cdist(batch, compare_batch, p=2).cpu()
            
            distances[start_row:end_row, start_col:end_col] = dists_batch
            if start_col != start_row:
                distances[start_col:end_col, start_row:end_row] = dists_batch.T
                
            del compare_batch, dists_batch
            torch.cuda.empty_cache()
        
        del batch
        torch.cuda.empty_cache()
    
    distances.fill_diagonal_(float('inf'))
    gc.collect()
    torch.cuda.empty_cache()
    return distances.numpy()

def _normalize_text(s: str) -> str:
    # lowercase
    s = s.lower()
    # strip accents
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # replace common separators with spaces, collapse whitespace
    s = re.sub(r'[\\/._\-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class CommonFilters:
    def __init__(self, engine, common_socket_events, media_directory, db_schema, update_model_ratings_func):
        self.engine = engine
        self.common_socket_events = common_socket_events
        self.media_directory = media_directory
        self.db_schema = db_schema
        self.update_model_ratings_func = update_model_ratings_func

        self.embedding_gathering_callback = EmbeddingGatheringCallback(self.common_socket_events.show_search_status, name="")
        self.meta_embedding_gathering_callback = EmbeddingGatheringCallback(self.common_socket_events.show_search_status, name="metadata")

        self.metadata_engine = MetadataSearch(self.engine.cfg)    

    def filter_by_file(self, all_files, text_query):
        target_path = text_query
        self.common_socket_events.show_search_status("Extracting embeddings")
        embeds_all = self.engine.process_files(all_files, callback=self.embedding_gathering_callback, media_folder=self.media_directory)
        target_emb = self.engine.process_files([target_path], callback=self.embedding_gathering_callback, media_folder=self.media_directory)

        self.common_socket_events.show_search_status("Computing distances between embeddings")
        dists = torch.cdist(embeds_all, target_emb, p=2).squeeze(-1)
        scores = (1.0 / (1.0 + dists)).cpu().detach().numpy()
        return scores

    def filter_by_text(self, all_files, text_query, **kwargs):
        mode = kwargs.get("mode", "file-name")

        if mode not in ('file-name', 'semantic-content', 'semantic-metadata'):
            raise ValueError(f"Unknown mode: {mode}")

        if mode == 'file-name':
            q = _normalize_text(text_query)

            # Prefer tokenized scorer when query has multiple words (order-insensitive)
            scorer = rapidfuzz.fuzz.token_set_ratio if ' ' in q else rapidfuzz.fuzz.WRatio
            # If strict reordering handling is desired, consider token_sort_ratio instead of token_set_ratio.

            ranked = []
            for p in all_files:
                base = os.path.basename(p)
                s_full = scorer(q, _normalize_text(p))
                s_base = scorer(q, _normalize_text(base))
                combined = max(1.2 * s_base, s_full)  # extra weight to basename
                ranked.append((combined, p, s_base, s_full))

            # Keep scores in original order (aligned with all_files)
            scores = np.array([r[0] for r in ranked], dtype=np.float32) / 100.0

        if mode == 'semantic-content':
            self.common_socket_events.show_search_status("Extracting embeddings")
            embeds_text = self.engine.process_text(text_query)
            embeds_files = self.engine.process_files(all_files, callback=self.embedding_gathering_callback, media_folder=self.media_directory)
            files_similarity_scores = self.engine.compare(embeds_files, embeds_text)
                
            self.common_socket_events.show_search_status("Sorting by relevance")
            scores = files_similarity_scores

        if mode == 'semantic-metadata':
            self.common_socket_events.show_search_status("Extracting metadata embeddings")
            embeds_meta_text = self.metadata_engine.process_query(text_query)
            embeds_meta_files = self.metadata_engine.process_files(all_files, callback=self.meta_embedding_gathering_callback, media_folder=self.media_directory)
            meta_similarity_scores = self.metadata_engine.compare(embeds_meta_files, embeds_meta_text)
                
            self.common_socket_events.show_search_status("Sorting by relevance")
            scores = meta_similarity_scores

        return scores

    def filter_by_file_size(self, all_files, text_query):
        scores = np.array([os.path.getsize(f) for f in all_files], dtype=np.float32)
        return scores

    def filter_by_random(self, all_files, text_query):
        scores = np.random.rand(len(all_files)).astype(np.float32)
        return scores

    def filter_by_rating(self, all_files, text_query):
        self.update_model_ratings_func(all_files)
        all_hashes = [self.engine.get_file_hash(f) for f in all_files]
        items = self.db_schema.query.filter(self.db_schema.hash.in_(all_hashes)).all()
        hash_to_rating = {item.hash: item.user_rating if item.user_rating is not None else item.model_rating for item in items}
        
        all_ratings = [hash_to_rating.get(h) for h in all_hashes]
        mean_score = np.mean([r for r in all_ratings if r is not None]) if any(r is not None for r in all_ratings) else 0
        all_ratings = [r if r is not None else mean_score for r in all_ratings]
        
        return all_ratings

    def filter_by_similarity(self, all_files, text_query):
        self.common_socket_events.show_search_status("Extracting embeddings for similarity sort")
        embeds = self.engine.process_files(all_files, callback=self.embedding_gathering_callback, media_folder=self.media_directory)
        if isinstance(embeds, torch.Tensor):
            embeds = embeds.cpu().detach().numpy()

        self.common_socket_events.show_search_status("Computing distances between embeddings")
        distances = compute_distances_batched(embeds)
        
        min_distances = np.min(distances, axis=1)
        target_indices = np.argmin(distances, axis=1)
        target_min_distances = min_distances[target_indices]
        file_sizes = [os.path.getsize(f) for f in all_files]

        files_with_metrics = list(zip(target_min_distances, min_distances, file_sizes, all_files))
        sorted_files_with_metrics = sorted(files_with_metrics, key=lambda x: (x[0], x[1], x[2]))
        
        # Convert ordering into scores (higher score = earlier in sorted list)
        max_rank = len(sorted_files_with_metrics)
        path_to_score = {
            f: (max_rank - rank)  # simple descending rank score
            for rank, (_, _, _, f) in enumerate(sorted_files_with_metrics)
        }
        scores = np.array([path_to_score[f] for f in all_files], dtype=np.float32)
        return scores