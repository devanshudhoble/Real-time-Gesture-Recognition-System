"""
ONNX Model Export Script
=========================
Exports the trained PyTorch gesture classifier to ONNX format for
optimized CPU inference on edge devices.

Features:
    - Exports to ONNX with proper input/output naming
    - Validates exported model with ONNX checker
    - Benchmarks PyTorch vs ONNX Runtime inference speed
    - Verifies numerical equivalence between backends

Usage:
    python training/export_onnx.py
"""

import time
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.classifier import GestureClassifier


def export_to_onnx():
    """Export the trained model to ONNX format."""
    print("=" * 60)
    print("ONNX Model Export")
    print("=" * 60)

    device = "cpu"

    # Load PyTorch model
    model = GestureClassifier.load_from_checkpoint(str(config.BEST_MODEL_PATH), device)
    model.eval()
    print(f"Loaded PyTorch model from: {config.BEST_MODEL_PATH}")
    print(f"Parameters: {model.count_parameters():,}")

    # Create dummy input
    input_dim = config.INPUT_FEATURE_DIM
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    # Export
    onnx_path = str(config.ONNX_MODEL_PATH)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"\nONNX model exported to: {onnx_path}")

    # Validate with ONNX checker
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validated successfully")

    # Check model size
    onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {onnx_size:.4f} MB")

    # ── Benchmark: PyTorch vs ONNX Runtime ──
    print(f"\n{'─' * 40}")
    print("Inference Speed Benchmark")
    print(f"{'─' * 40}")

    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    test_input = np.random.randn(1, input_dim).astype(np.float32)
    test_tensor = torch.tensor(test_input)

    n_iterations = 1000

    # PyTorch benchmark
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(50):
            _ = model(test_tensor)

        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = model(test_tensor)
        pytorch_time = (time.perf_counter() - start) / n_iterations * 1000

    # ONNX Runtime benchmark
    # Warmup
    for _ in range(50):
        _ = ort_session.run(None, {input_name: test_input})

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = ort_session.run(None, {input_name: test_input})
    onnx_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"PyTorch inference:      {pytorch_time:.4f} ms/sample")
    print(f"ONNX Runtime inference: {onnx_time:.4f} ms/sample")
    print(f"Speedup:                {pytorch_time / onnx_time:.2f}x")

    # ── Verify numerical equivalence ──
    print(f"\n{'─' * 40}")
    print("Numerical Equivalence Check")
    print(f"{'─' * 40}")

    with torch.no_grad():
        pytorch_output = model(test_tensor).numpy()

    onnx_output = ort_session.run(None, {input_name: test_input})[0]

    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"Max difference: {max_diff:.8f}")
    if max_diff < 1e-5:
        print("✓ Outputs match within tolerance")
    else:
        print("⚠ Warning: Outputs differ more than expected")

    # ── Batch inference benchmark ──
    print(f"\n{'─' * 40}")
    print("Batch Inference Benchmark")
    print(f"{'─' * 40}")

    for batch_size in [1, 8, 32]:
        batch_input = np.random.randn(batch_size, input_dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(100):
            _ = ort_session.run(None, {input_name: batch_input})
        batch_time = (time.perf_counter() - start) / 100 * 1000
        per_sample = batch_time / batch_size

        print(f"  Batch={batch_size:3d}: {batch_time:.4f} ms total, {per_sample:.4f} ms/sample")

    print("\n" + "=" * 60)
    print("ONNX export complete!")
    print("=" * 60)


if __name__ == "__main__":
    export_to_onnx()
