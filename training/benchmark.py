"""
Runtime Benchmark Script
========================
Measures end-to-end webcam inference performance and saves the results as
machine-readable JSON for submission artifacts.

Usage:
    python training/benchmark.py
    python training/benchmark.py --onnx --seconds 20
"""

import argparse
import json
import time
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.pipeline import GestureRecognitionPipeline


def open_camera(index: int, width: int, height: int):
    """Open a webcam with fallback indices and backends."""
    indices_to_try = [index] + [i for i in range(5) if i != index]

    for idx in indices_to_try:
        for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)

            success_count = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    success_count += 1
                time.sleep(0.05)

            if success_count >= 2:
                return cap, idx

            cap.release()

    raise RuntimeError(f"Could not open camera index {index} or any fallback index")


def benchmark(args):
    """Run a timed benchmark and save runtime metrics."""
    backend_label = "ONNX" if args.onnx else "PyTorch"
    print("=" * 60)
    print("Gesture Recognition Runtime Benchmark")
    print("=" * 60)
    print(f"Backend: {backend_label}")
    print(f"Camera : {args.camera}")
    print(f"Window : {args.seconds:.1f} seconds")

    pipeline = GestureRecognitionPipeline(
        model_path=args.model,
        use_onnx=args.onnx,
    )
    cap, actual_camera_idx = open_camera(args.camera, args.width, args.height)

    start_time = time.perf_counter()
    frames = 0
    total_detected_hands = 0

    try:
        while (time.perf_counter() - start_time) < args.seconds:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            results = pipeline.process_frame(frame)
            frames += 1
            total_detected_hands += len(results)
    finally:
        cap.release()
        pipeline.release()

    elapsed = time.perf_counter() - start_time
    latency_stats = pipeline.get_latency_stats()
    metrics = {
        "backend": backend_label,
        "camera_index": actual_camera_idx,
        "duration_seconds": elapsed,
        "frames_processed": frames,
        "effective_fps": (frames / elapsed) if elapsed > 0 else 0.0,
        "pipeline_fps": pipeline.get_fps(),
        "average_hands_per_frame": (total_detected_hands / frames) if frames > 0 else 0.0,
        "latency_ms": latency_stats,
        "frame_size": {
            "width": args.width,
            "height": args.height,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFrames processed : {frames}")
    print(f"Effective FPS    : {metrics['effective_fps']:.2f}")
    print(f"Pipeline FPS     : {metrics['pipeline_fps']:.2f}")
    print(f"Mean latency     : {latency_stats.get('mean', 0.0):.2f} ms")
    print(f"P95 latency      : {latency_stats.get('p95', 0.0):.2f} ms")
    print(f"Saved metrics to : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gesture recognition runtime")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--onnx", action="store_true", help="Use ONNX Runtime backend")
    parser.add_argument("--model", type=str, default=None, help="Override model file path")
    parser.add_argument("--seconds", type=float, default=15.0, help="Benchmark duration in seconds")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal frame flip")
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.MODELS_DIR / "runtime_metrics.json"),
        help="Path to save benchmark metrics JSON",
    )
    benchmark(parser.parse_args())
