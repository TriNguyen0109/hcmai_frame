import numpy as np
from typing import List, Dict, Any, Tuple


class IntraSceneDeduplicator:
    """
    Step 4: Storage Optimization / Near-duplicate Deduplication.
    Filter keyframes within the same scene if Cosine Similarity > threshold (default 0.9).
    """

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold

    def deduplicate_scene_keyframes(self,
                                   keyframes: List[Dict[str, Any]],
                                   clip_vectors: np.ndarray,
                                   beit_vectors: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Calculates cosine similarity matrix between keyframes within a single scene.
        Drops keyframe j if similarity(i, j) > similarity_threshold for any kept i < j.
        Returns filtered keyframe objects, filtered clip_vectors, filtered beit_vectors.
        """
        n = len(keyframes)
        if n <= 1:
            return keyframes, clip_vectors, beit_vectors

        # Cosine similarity matrix (vectors are already L2 normalized)
        # sim_matrix[i, j] = dot(v_i, v_j)
        sim_matrix = np.dot(clip_vectors, clip_vectors.T)

        keep_indices = []
        for i in range(n):
            is_duplicate = False
            for prev_idx in keep_indices:
                if sim_matrix[prev_idx, i] > self.similarity_threshold:
                    is_duplicate = True
                    print(f"[Deduplicator] Scene {keyframes[i].get('scene_index')}: "
                          f"Removing near-duplicate {keyframes[i]['frame_id']} "
                          f"(Cosine Similarity {sim_matrix[prev_idx, i]:.4f} > {self.similarity_threshold})")
                    break
            if not is_duplicate:
                keep_indices.append(i)

        filtered_keyframes = [keyframes[i] for i in keep_indices]
        filtered_clip = clip_vectors[keep_indices]
        filtered_beit = beit_vectors[keep_indices]

        return filtered_keyframes, filtered_clip, filtered_beit
