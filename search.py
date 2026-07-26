import os
import json
import argparse
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any


class VideoSearchEngine:
    """
    Query & Search Engine for the indexed video database.
    Performs multimodal text/image search against FAISS indexes and returns matching scenes/keyframes.
    """

    def __init__(self, storage_dir: str = "./storage", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.storage_dir = storage_dir
        self.device = device

        clip_idx_path = os.path.join(storage_dir, "index_clip.index")
        beit_idx_path = os.path.join(storage_dir, "index_beit3.index")

        if not os.path.exists(clip_idx_path) or not os.path.exists(beit_idx_path):
            raise FileNotFoundError(f"FAISS index files not found in {storage_dir}. Run pipeline.py first.")

        self.index_clip = faiss.read_index(clip_idx_path)
        self.index_beit3 = faiss.read_index(beit_idx_path)

        # Load Metadata JSONs
        self.metadata_lookup = {}
        for fname in os.listdir(storage_dir):
            if fname.endswith("_metadata.json"):
                path = os.path.join(storage_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    video_id = meta["video_id"]
                    for scene in meta["scenes"]:
                        for kf in scene["keyframes"]:
                            vec_id = kf["clip_vector_id"]
                            self.metadata_lookup[vec_id] = {
                                "video_id": video_id,
                                "scene_id": scene["scene_id"],
                                "start_time": scene["start_time"],
                                "end_time": scene["end_time"],
                                "frame_id": kf["frame_id"],
                                "timestamp": kf["timestamp"],
                                "image_path": kf["image_path"]
                            }

        # Initialize CLIP model for query embedding
        try:
            from transformers import CLIPProcessor, CLIPModel
            print(f"[SearchEngine] Loading CLIP query encoder...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_model.eval()
        except Exception as e:
            print(f"[SearchEngine] Error loading CLIP query encoder: {e}")
            self.clip_model = None

    @torch.no_grad()
    def embed_text_query(self, text: str) -> np.ndarray:
        """
        Embeds text query into CLIP vector space.
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP model is not initialized.")

        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
            text_features = text_features.pooler_output
        elif hasattr(text_features, "text_embeds"):
            text_features = text_features.text_embeds
        elif isinstance(text_features, (tuple, list)):
            text_features = text_features[0]
        text_features = F.normalize(text_features, p=2, dim=-1)
        arr = text_features.cpu().numpy().astype(np.float32)
        return arr.reshape(1, -1)

    def search_by_text(self, text_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the video database using a natural language text query.
        """
        print(f"\n[SearchEngine] Searching for: '{text_query}' (Top {top_k})...")
        query_vector = self.embed_text_query(text_query)

        # Search in CLIP index
        scores, indices = self.index_clip.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Vector ID maps to index + 1001 (based on storage manager offset)
            vector_id = idx + 1001
            meta = self.metadata_lookup.get(vector_id, {})
            results.append({
                "score": float(score),
                "vector_id": vector_id,
                **meta
            })

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Indexing Search Engine CLI")
    parser.add_argument("--query", type=str, required=True, help="Text query to search in indexed videos")
    parser.add_argument("--storage_dir", type=str, default="./storage", help="Storage directory containing FAISS index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top search results to return")

    args = parser.parse_args()

    engine = VideoSearchEngine(storage_dir=args.storage_dir)
    results = engine.search_by_text(text_query=args.query, top_k=args.top_k)

    print("\n" + "=" * 60)
    print(f"SEARCH RESULTS FOR: '{args.query}'")
    print("=" * 60)
    for i, res in enumerate(results, start=1):
        print(f"\nTop {i} (Cosine Sim: {res['score']:.4f}):")
        print(f"  Video ID  : {res.get('video_id')}")
        print(f"  Scene ID  : {res.get('scene_id')} ({res.get('start_time')}s -> {res.get('end_time')}s)")
        print(f"  Timestamp : {res.get('timestamp')}s")
        print(f"  Frame ID  : {res.get('frame_id')}")
        print(f"  Image Path: {res.get('image_path')}")
    print("=" * 60)
