import os
import torch
import numpy as np
import gc

from pages.utils import SortingProgressCallback, EmbeddingGatheringCallback

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

class CommonFilters:
    def __init__(self, engine, common_socket_events, media_directory, db_schema, update_model_ratings_func):
        self.engine = engine
        self.common_socket_events = common_socket_events
        self.media_directory = media_directory
        self.db_schema = db_schema
        self.update_model_ratings_func = update_model_ratings_func
        self.embedding_gathering_callback = EmbeddingGatheringCallback(self.common_socket_events.show_search_status)

    def filter_by_file(self, all_files, text_query):
        target_path = text_query
        self.common_socket_events.show_search_status("Extracting embeddings")
        embeds_all = self.engine.process_files(all_files, callback=self.embedding_gathering_callback, media_folder=self.media_directory)
        target_emb = self.engine.process_files([target_path], callback=self.embedding_gathering_callback, media_folder=self.media_directory)

        self.common_socket_events.show_search_status("Computing distances between embeddings")
        scores = torch.cdist(embeds_all, target_emb, p=2).cpu().detach().numpy()
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__)
        return [all_files[i] for i in sorted_indices]

    def filter_by_text(self, all_files, text_query):
        self.common_socket_events.show_search_status("Extracting embeddings")
        embeds_files = self.engine.process_files(all_files, callback=self.embedding_gathering_callback, media_folder=self.media_directory)
        embeds_text = self.engine.process_text(text_query)
        scores = self.engine.compare(embeds_files, embeds_text)

        self.common_socket_events.show_search_status("Sorting by relevance")
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [all_files[i] for i in sorted_indices]

    def filter_by_file_size(self, all_files, text_query):
        return sorted(all_files, key=os.path.getsize)

    def filter_by_random(self, all_files, text_query):
        shuffled_indices = np.random.permutation(len(all_files))
        return [all_files[i] for i in shuffled_indices]

    def filter_by_rating(self, all_files, text_query):
        self.update_model_ratings_func(all_files)
        all_hashes = [self.engine.cached_file_hash.get_file_hash(f) for f in all_files]
        items = self.db_schema.query.filter(self.db_schema.hash.in_(all_hashes)).all()
        hash_to_rating = {item.hash: item.user_rating if item.user_rating is not None else item.model_rating for item in items}
        
        all_ratings = [hash_to_rating.get(h) for h in all_hashes]
        mean_score = np.mean([r for r in all_ratings if r is not None]) if any(r is not None for r in all_ratings) else 0
        all_ratings = [r if r is not None else mean_score for r in all_ratings]
        
        return [file for _, file in sorted(zip(all_ratings, all_files), reverse=True)]

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
        
        return [f for _, _, _, f in sorted_files_with_metrics]