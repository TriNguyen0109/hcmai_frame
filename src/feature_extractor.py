import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple


class DualFeatureExtractor:
    """
    Step 3: Dual Feature Extraction using CLIP and BEiT/BEiT-3 models.
    """

    def __init__(self,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 beit_model_name: str = "microsoft/beit-base-patch16-224",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[DualFeatureExtractor] Initializing on device: {self.device}")

        # 1. Initialize CLIP Model & Processor
        try:
            from transformers import CLIPProcessor, CLIPModel
            print(f"[DualFeatureExtractor] Loading CLIP model ({clip_model_name})...")
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            self.clip_model.eval()
        except Exception as e:
            print(f"[DualFeatureExtractor] Error loading CLIP model ({e}). Using mock CLIP extractor.")
            self.clip_model = None

        # 2. Initialize BEiT / BEiT-3 Model & Processor
        try:
            from transformers import AutoImageProcessor, AutoModel
            print(f"[DualFeatureExtractor] Loading BEiT model ({beit_model_name})...")
            self.beit_processor = AutoImageProcessor.from_pretrained(beit_model_name)
            self.beit_model = AutoModel.from_pretrained(beit_model_name).to(self.device)
            self.beit_model.eval()
        except Exception as e:
            print(f"[DualFeatureExtractor] Error loading BEiT model ({e}). Using mock BEiT extractor.")
            self.beit_model = None

    @torch.no_grad()
    def extract_features(self, pil_images: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts L2-normalized CLIP and BEiT feature vectors for a list of PIL images.
        Returns:
            clip_vectors: np.ndarray [N, d_clip]
            beit_vectors: np.ndarray [N, d_beit]
        """
        if len(pil_images) == 0:
            return np.array([]), np.array([])

        # --- CLIP Feature Extraction ---
        if self.clip_model is not None:
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.device)
            clip_features = self.clip_model.get_image_features(**clip_inputs)
            clip_features = F.normalize(clip_features, p=2, dim=1)
            clip_vectors = clip_features.cpu().numpy().astype(np.float32)
        else:
            # Fallback mock random normalized vectors (512-dim)
            mock_vecs = np.random.randn(len(pil_images), 512).astype(np.float32)
            clip_vectors = mock_vecs / np.linalg.norm(mock_vecs, axis=1, keepdims=True)

        # --- BEiT Feature Extraction ---
        if self.beit_model is not None:
            beit_inputs = self.beit_processor(images=pil_images, return_tensors="pt").to(self.device)
            outputs = self.beit_model(**beit_inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                beit_features = outputs.pooler_output
            else:
                beit_features = outputs.last_hidden_state[:, 0, :]
            beit_features = F.normalize(beit_features, p=2, dim=1)
            beit_vectors = beit_features.cpu().numpy().astype(np.float32)
        else:
            # Fallback mock random normalized vectors (768-dim)
            mock_vecs = np.random.randn(len(pil_images), 768).astype(np.float32)
            beit_vectors = mock_vecs / np.linalg.norm(mock_vecs, axis=1, keepdims=True)

        return clip_vectors, beit_vectors
