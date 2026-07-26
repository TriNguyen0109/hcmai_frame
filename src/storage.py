import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any


class IndexStorageManager:
    """
    Step 5: Storage & FAISS Indexing.
    Manages FAISS indices (index_clip, index_beit3) and saves metadata JSON.
    """

    def __init__(self, clip_dim: int = 512, beit_dim: int = 768, storage_dir: str = "./storage"):
        self.clip_dim = clip_dim
        self.beit_dim = beit_dim
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        # FAISS IndexFlatIP (Inner Product = Cosine Similarity for L2-normalized vectors)
        self.index_clip = faiss.IndexFlatIP(self.clip_dim)
        self.index_beit3 = faiss.IndexFlatIP(self.beit_dim)

        self.current_vector_id = 1000

    def add_keyframes_and_save(self,
                              video_id: str,
                              scenes_with_keyframes: List[Dict[str, Any]],
                              all_clip_vectors: np.ndarray,
                              all_beit_vectors: np.ndarray) -> str:
        """
        Adds vectors to FAISS indices and writes the final metadata JSON file.
        """
        vec_idx = 0

        metadata = {
            "video_id": video_id,
            "scenes": []
        }

        for scene in scenes_with_keyframes:
            scene_entry = {
                "scene_id": scene["scene_id"],
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "keyframes": []
            }

            for kf in scene["keyframes"]:
                # Increment vector ID
                self.current_vector_id += 1
                v_id = self.current_vector_id

                clip_vec = all_clip_vectors[vec_idx:vec_idx+1]
                beit_vec = all_beit_vectors[vec_idx:vec_idx+1]
                vec_idx += 1

                # Add to FAISS index
                self.index_clip.add(clip_vec)
                self.index_beit3.add(beit_vec)

                # Clean image_path for JSON metadata output
                img_path = kf["image_path"].replace("\\", "/")

                scene_entry["keyframes"].append({
                    "frame_id": kf["frame_id"],
                    "timestamp": kf["timestamp"],
                    "image_path": img_path,
                    "clip_vector_id": v_id,
                    "beit3_vector_id": v_id
                })

            metadata["scenes"].append(scene_entry)

        # Save JSON metadata
        json_path = os.path.join(self.storage_dir, f"{video_id}_metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save FAISS Index files
        clip_index_path = os.path.join(self.storage_dir, "index_clip.index")
        beit_index_path = os.path.join(self.storage_dir, "index_beit3.index")

        faiss.write_index(self.index_clip, clip_index_path)
        faiss.write_index(self.index_beit3, beit_index_path)

        print(f"[StorageManager] Metadata saved to: {json_path}")
        print(f"[StorageManager] FAISS CLIP Index saved to: {clip_index_path} (Total: {self.index_clip.ntotal})")
        print(f"[StorageManager] FAISS BEiT3 Index saved to: {beit_index_path} (Total: {self.index_beit3.ntotal})")

        return json_path
