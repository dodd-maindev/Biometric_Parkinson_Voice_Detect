"""CLI script to train and evaluate the Acoustic Domain Router and Multi-Domain Pipeline."""

import argparse
from pathlib import Path
import numpy as np
from src.application.pipelines.multi_domain_routing_pipeline import MultiDomainRoutingPipeline
from src.infrastructure.models.acoustic_domain_router import AcousticDomainRouter


def main() -> None:
    """Parse CLI arguments and run router training / multi-domain evaluation."""
    parser = argparse.ArgumentParser(description="Acoustic Domain Router Training & Evaluation")
    parser.add_argument("--kcl_features", type=str, default="./data/processed/mdvr/features_acoustic_gtcc.npz")
    parser.add_argument("--voice_features", type=str, default="./data/processed/voice_dataset/features_acoustic_gtcc.npz")
    parser.add_argument("--kcl_model", type=str, default="/content/drive/MyDrive/parkinson_tri_modal_97_44_best.joblib")
    parser.add_argument("--voice_model", type=str, default="/content/drive/MyDrive/parkinson_tri_modal_voice_dataset_best.joblib")
    parser.add_argument("--output_router_path", type=str, default="/content/drive/MyDrive/domain_router_best.joblib")
    args = parser.parse_args()

    kcl_feat_path = Path(args.kcl_features)
    voice_feat_path = Path(args.voice_features)
    if not kcl_feat_path.exists() or not voice_feat_path.exists():
        return print(f"Feature caches not found! Ensure baseline/tri_modal has run for both datasets.")

    print(f"Loading acoustic features from:\n  - KCL: {kcl_feat_path}\n  - Voice: {voice_feat_path}")
    kcl_data = np.load(kcl_feat_path)
    voice_data = np.load(voice_feat_path)
    feat_kcl, feat_voice = kcl_data["features"], voice_data["features"]

    # Filter invalid pitch if present
    feat_kcl = feat_kcl[feat_kcl[:, 0] > 0.0]
    feat_voice = feat_voice[feat_voice[:, 0] > 0.0]

    router_path = Path(args.output_router_path)
    pipeline = MultiDomainRoutingPipeline(Path(args.kcl_model), Path(args.voice_model), router_path)
    pipeline.train_router(feat_kcl, feat_voice, router_path)

    # Evaluate router on test splits or full domains
    router = AcousticDomainRouter.load(router_path)
    pred_kcl = router.predict_domain(feat_kcl)
    pred_voice = router.predict_domain(feat_voice)
    acc_kcl = np.mean(pred_kcl == 0) * 100
    acc_voice = np.mean(pred_voice == 1) * 100
    print("\n" + "=" * 55)
    print("===== DOMAIN ROUTER ACCURACY =====")
    print(f"  MDVR-KCL -> Domain 0 (KCL Model):       {acc_kcl:.2f}%")
    print(f"  Voice_Dataset -> Domain 1 (Voice Model): {acc_voice:.2f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
