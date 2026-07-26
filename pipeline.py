import os
import argparse
import numpy as np
from src.scene_detector import SceneDetector
from src.keyframe_extractor import KeyframeExtractor
from src.feature_extractor import DualFeatureExtractor
from src.deduplicator import IntraSceneDeduplicator
from src.storage import IndexStorageManager


def run_indexing_pipeline(video_path: str,
                          output_dir: str = "./storage",
                          device: str = "cuda",
                          sim_threshold: float = 0.9):
    """
    Runs the complete 5-step video indexing pipeline.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]

    print("=" * 70)
    print(f"STARTING VIDEO INDEXING PIPELINE FOR: {video_id}")
    print("=" * 70)

    # Step 1: Scene Boundary Detection
    print("\n[Step 1/5] Scene Boundary Detection with TransNetV2...")
    scene_detector = SceneDetector(device=device)
    raw_scenes = scene_detector.detect_scenes(video_path)
    print(f" -> Detected {len(raw_scenes)} scenes.")

    # Step 2: Keyframe Selection (4 evenly spaced keyframes)
    print("\n[Step 2/5] Extracting 4 keyframes per scene...")
    keyframe_extractor = KeyframeExtractor(num_keyframes=4)
    scenes_with_keyframes = keyframe_extractor.extract_keyframes(
        video_path=video_path,
        scenes=raw_scenes,
        output_dir=output_dir,
        video_id=video_id
    )

    # Step 3 & 4: Feature Extraction & Deduplication
    print("\n[Step 3/5 & Step 4/5] Extracting CLIP & BEiT-3 features and Deduplicating...")
    feature_extractor = DualFeatureExtractor(device=device)
    deduplicator = IntraSceneDeduplicator(similarity_threshold=sim_threshold)

    final_scenes = []
    all_final_clip_vecs = []
    all_final_beit_vecs = []

    total_extracted = 0
    total_retained = 0

    for scene in scenes_with_keyframes:
        keyframes = scene["keyframes"]
        if len(keyframes) == 0:
            continue

        total_extracted += len(keyframes)
        pil_images = [kf["pil_image"] for kf in keyframes]

        # Extract features (Step 3)
        clip_vecs, beit_vecs = feature_extractor.extract_features(pil_images)

        # Deduplicate (Step 4)
        filtered_kfs, filtered_clip, filtered_beit = deduplicator.deduplicate_scene_keyframes(
            keyframes=keyframes,
            clip_vectors=clip_vecs,
            beit_vectors=beit_vecs
        )

        total_retained += len(filtered_kfs)

        scene_copy = dict(scene)
        scene_copy["keyframes"] = filtered_kfs
        final_scenes.append(scene_copy)

        if len(filtered_clip) > 0:
            all_final_clip_vecs.append(filtered_clip)
            all_final_beit_vecs.append(filtered_beit)

    print(f" -> Keyframe Deduplication Summary: Retained {total_retained}/{total_extracted} keyframes.")

    # Step 5: FAISS Storage & Metadata Export
    print("\n[Step 5/5] Indexing vectors into FAISS and writing Metadata JSON...")
    if len(all_final_clip_vecs) > 0:
        stacked_clip = np.vstack(all_final_clip_vecs)
        stacked_beit = np.vstack(all_final_beit_vecs)
    else:
        stacked_clip = np.empty((0, 512), dtype=np.float32)
        stacked_beit = np.empty((0, 768), dtype=np.float32)

    clip_dim = stacked_clip.shape[1] if len(stacked_clip) > 0 else 512
    beit_dim = stacked_beit.shape[1] if len(stacked_beit) > 0 else 768

    storage_manager = IndexStorageManager(clip_dim=clip_dim, beit_dim=beit_dim, storage_dir=output_dir)
    json_metadata_path = storage_manager.add_keyframes_and_save(
        video_id=video_id,
        scenes_with_keyframes=final_scenes,
        all_clip_vectors=stacked_clip,
        all_beit_vectors=stacked_beit
    )

    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Metadata output file: {json_metadata_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Indexing Pipeline Baseline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="./storage", help="Output directory for metadata and FAISS index")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run models on (cuda or cpu)")
    parser.add_argument("--sim_threshold", type=float, default=0.9, help="Cosine similarity threshold for near-duplicate filtering")

    args = parser.parse_args()
    run_indexing_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        device=args.device,
        sim_threshold=args.sim_threshold
    )
