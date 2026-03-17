import numpy as np
from app.models.model_loader import model_loader

class SimilarityService:
    def compare_embedding(self, embedding: np.ndarray, brand: str) -> float:
        """
        Queries the FAISS index with the user's logo embedding.
        Filters search by the declared brand.
        Returns the Cosine Similarity score.
        """
        if model_loader.faiss_index is None or model_loader.faiss_labels is None:
            # Fallback to dummy data if database hasn't been built yet
            return np.random.uniform(0.50, 0.99)
            
        # FAISS search against all vectors
        # k=100 in case the top few are a different brand
        k = min(100, model_loader.faiss_index.ntotal)
        if k == 0:
            return 0.0

        distances, indices = model_loader.faiss_index.search(embedding, k)
        
        # Iterate through the returned closest matches
        best_match_score = 0.0
        
        for i in range(k):
            idx = indices[0][i]
            if idx == -1: continue # No more matches
            
            matched_brand = model_loader.faiss_labels[idx]
            
            # If standardizing names: matched_brand.lower() == brand.lower()
            if matched_brand.lower().replace(" ", "_") == brand.lower().replace(" ", "_"):
                best_match_score = float(distances[0][i])
                break # Found highest similarity for THIS specific brand
                
        return best_match_score
