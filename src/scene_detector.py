import os
import sys
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

# Add TransNetV2 path to sys.path if present
TRANSNETV2_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TransNetV2")
if os.path.exists(TRANSNETV2_PATH) and TRANSNETV2_PATH not in sys.path:
    sys.path.insert(0, TRANSNETV2_PATH)


class SceneDetector:
    """
    Step 1: Scene Boundary Detection using TransNetV2 with OpenCV/Histogram fallback.
    """

    def __init__(self, threshold: float = 0.5, weights_dir: str = None, device: str = "cuda"):
        self.threshold = threshold
        self.device = device
        self.tf_model = None
        self.pt_model = None
        self.engine_type = "fallback"

        if weights_dir is None:
            weights_dir = os.path.join(TRANSNETV2_PATH, "inference", "transnetv2-weights")

        # Attempt 1: Try loading TensorFlow TransNetV2
        try:
            from inference.transnetv2 import TransNetV2 as TFTransNetV2
            if os.path.exists(weights_dir):
                self.tf_model = TFTransNetV2(model_dir=weights_dir)
                self.engine_type = "tensorflow"
                print(f"[SceneDetector] TransNetV2 loaded successfully using TensorFlow engine.")
        except Exception as e:
            print(f"[SceneDetector] TensorFlow TransNetV2 not available ({e}). Trying PyTorch...")

        # Attempt 2: Try PyTorch TransNetV2 if TensorFlow was not loaded
        if self.engine_type == "fallback":
            try:
                import torch
                pt_weights_path = os.path.join(TRANSNETV2_PATH, "inference-pytorch", "transnetv2-pytorch-weights.pth")
                if os.path.exists(pt_weights_path):
                    from inference_pytorch.transnetv2_pytorch import TransNetV2 as PTTransNetV2
                    state_dict = torch.load(pt_weights_path, map_location=device)
                    self.pt_model = PTTransNetV2().to(device)
                    self.pt_model.load_state_dict(state_dict)
                    self.pt_model.eval()
                    self.engine_type = "pytorch"
                    print(f"[SceneDetector] TransNetV2 loaded successfully using PyTorch engine.")
            except Exception as e:
                print(f"[SceneDetector] PyTorch TransNetV2 not loaded ({e}).")

        if self.engine_type == "fallback":
            print("[SceneDetector] Warning: Using OpenCV Histogram Difference fallback for scene detection.")

    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detects scene boundaries in a video.
        Returns a list of dicts:
        [
            {
                "scene_index": 0,
                "start_frame": 0,
                "end_frame": 120,
                "start_time": 0.0,
                "end_time": 4.0
            }, ...
        ]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        raw_scenes = []

        if self.engine_type == "tensorflow" and self.tf_model is not None:
            try:
                video_frames, single_frame_pred, all_frame_pred = self.tf_model.predict_video(video_path)
                predictions = all_frame_pred
                raw_scenes = self.tf_model.predictions_to_scenes(predictions, threshold=self.threshold)
            except Exception as e:
                print(f"[SceneDetector] TF prediction failed ({e}), falling back to OpenCV...")
                raw_scenes = self._fallback_opencv_scenes(video_path, total_frames)
        else:
            raw_scenes = self._fallback_opencv_scenes(video_path, total_frames)

        scenes_metadata = []
        for idx, (s_frame, e_frame) in enumerate(raw_scenes):
            scenes_metadata.append({
                "scene_index": idx,
                "start_frame": int(s_frame),
                "end_frame": int(e_frame),
                "start_time": round(float(s_frame) / fps, 3),
                "end_time": round(float(e_frame) / fps, 3)
            })

        return scenes_metadata

    def _fallback_opencv_scenes(self, video_path: str, total_frames: int, threshold: float = 0.35) -> np.ndarray:
        """
        OpenCV HSV Histogram difference fallback for scene cut detection.
        """
        cap = cv2.VideoCapture(video_path)
        prev_hist = None
        cuts = [0]
        curr_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_hist is not None:
                sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if (1.0 - sim) > threshold:
                    cuts.append(curr_frame)

            prev_hist = hist
            curr_frame += 1

        cap.release()

        scenes = []
        for i in range(len(cuts)):
            start = cuts[i]
            end = cuts[i + 1] - 1 if i + 1 < len(cuts) else total_frames - 1
            if end >= start:
                scenes.append([start, end])

        if len(scenes) == 0:
            scenes = [[0, max(0, total_frames - 1)]]

        return np.array(scenes, dtype=np.int32)
