import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any


class KeyframeExtractor:
    """
    Step 2: Keyframe Selection.
    Selects N evenly spaced keyframes (default 4) for each detected scene.
    """

    def __init__(self, num_keyframes: int = 4):
        self.num_keyframes = num_keyframes

    def extract_keyframes(self, video_path: str, scenes: List[Dict[str, Any]], output_dir: str, video_id: str) -> List[Dict[str, Any]]:
        """
        Extracts keyframes from video based on scene boundaries.
        Saves keyframe images to disk under output_dir/keyframes/{video_id}/.
        Returns scene list updated with keyframes data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        keyframe_dir = os.path.join(output_dir, "keyframes", video_id)
        os.makedirs(keyframe_dir, exist_ok=True)

        processed_scenes = []

        for scene in scenes:
            scene_index = scene["scene_index"]
            start_frame = scene["start_frame"]
            end_frame = scene["end_frame"]

            # Generate 4 evenly spaced frame indices
            if end_frame > start_frame:
                frame_indices = np.linspace(start_frame, end_frame, self.num_keyframes, dtype=int)
            else:
                frame_indices = np.full(self.num_keyframes, start_frame, dtype=int)

            keyframes_data = []
            for kf_idx, f_idx in enumerate(frame_indices, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                frame_id = f"frame_{video_id}_s{scene_index:03d}_{kf_idx}"
                img_filename = f"{frame_id}.jpg"
                img_path = os.path.join(keyframe_dir, img_filename)
                
                # Save image file
                cv2.imwrite(img_path, frame_bgr)

                timestamp = round(float(f_idx) / fps, 3)

                keyframes_data.append({
                    "frame_id": frame_id,
                    "frame_index": int(f_idx),
                    "timestamp": timestamp,
                    "image_path": img_path,
                    "pil_image": pil_img,
                    "scene_index": scene_index
                })

            scene_copy = dict(scene)
            scene_copy["scene_id"] = f"scene_{video_id}_{scene_index:03d}"
            scene_copy["keyframes"] = keyframes_data
            processed_scenes.append(scene_copy)

        cap.release()
        return processed_scenes
