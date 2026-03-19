"""
run_all.py — Gesture Recognition System: Easy Launcher
=========================================================
Runs the full pipeline in one command, from dataset check → training → evaluation → demo.

Usage (from project root):
    python run_all.py           # Complete pipeline
    python run_all.py --demo    # Jump straight to live demo (skip training)
    python run_all.py --train   # Only retrain (won't open demo)
    python run_all.py --eval    # Only evaluate + print metrics
    python run_all.py --benchmark  # Run non-interactive runtime benchmark
"""

import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list, description: str):
    """Run a subprocess command, print output, and check for errors."""
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"{'─'*60}")
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with code {result.returncode}")
        sys.exit(result.returncode)


def check_dataset():
    """Check if dataset exists, generate if missing."""
    features_path = ROOT / "data" / "features.npy"
    labels_path   = ROOT / "data" / "labels.npy"
    if features_path.exists() and labels_path.exists():
        import numpy as np
        n = np.load(features_path).shape[0]
        print(f"  ✓ Dataset already exists ({n} samples)")
        return
    print("  Dataset not found — generating...")
    run(["data/generate_dataset.py"], "Generating synthetic dataset")


def check_model():
    """Return True if trained model weights already exist."""
    model_path = ROOT / "models" / "gesture_classifier_best.pth"
    return model_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Gesture Recognition — Full Pipeline Launcher")
    parser.add_argument("--demo",  action="store_true", help="Skip to live demo immediately")
    parser.add_argument("--train", action="store_true", help="Only retrain the model")
    parser.add_argument("--eval",  action="store_true", help="Only evaluate the model")
    parser.add_argument("--benchmark", action="store_true", help="Only run runtime benchmark")
    parser.add_argument("--onnx",  action="store_true", help="Use ONNX backend in demo")
    parser.add_argument("--camera",type=int, default=0,  help="Camera index for demo")
    parser.add_argument("--data", type=str, default=None, help="Dataset directory for training")
    parser.add_argument("--seconds", type=float, default=15.0, help="Benchmark duration in seconds")
    args = parser.parse_args()

    print("=" * 60)
    print("  🤚 Gesture Recognition System — Pipeline Launcher")
    print("=" * 60)

    if args.demo:
        # Jump straight to demo (assumes model already trained)
        demo_cmd = ["demo/demo.py", f"--camera={args.camera}"]
        if args.onnx:
            demo_cmd.append("--onnx")
        run(demo_cmd, "Running live demo (press 'q' to quit)")
        return

    if args.eval:
        run(["training/evaluate.py"], "Evaluating model on test set")
        return

    if args.benchmark:
        benchmark_cmd = [
            "training/benchmark.py",
            f"--camera={args.camera}",
            f"--seconds={args.seconds}",
        ]
        if args.onnx:
            benchmark_cmd.append("--onnx")
        run(benchmark_cmd, "Benchmarking runtime performance")
        return

    # ── Full pipeline ──
    if args.data is None:
        check_dataset()

    if args.train or not check_model():
        train_cmd = ["training/train.py"]
        if args.data:
            train_cmd.append(f"--data={args.data}")
        run(train_cmd, "Training gesture classifier")
    else:
        print(f"\n  ✓ Model already trained, skipping training step")
        print(f"    (use --train to force retrain)")

    run(["training/evaluate.py"], "Evaluating on held-out test set")

    try:
        run(["training/export_onnx.py"], "Exporting to ONNX format")
    except SystemExit:
        print("  (ONNX export failed — skipping, PyTorch backend will be used)")

    benchmark_cmd = [
        "training/benchmark.py",
        f"--camera={args.camera}",
        f"--seconds={args.seconds}",
    ]
    if args.onnx:
        benchmark_cmd.append("--onnx")
    run(benchmark_cmd, "Benchmarking runtime performance")

    demo_cmd = ["demo/demo.py", f"--camera={args.camera}"]
    if args.onnx:
        demo_cmd.append("--onnx")
    run(demo_cmd, "Launching live webcam demo (press 'q' to quit)")


if __name__ == "__main__":
    main()
