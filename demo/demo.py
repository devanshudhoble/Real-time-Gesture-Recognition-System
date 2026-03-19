"""
Real-Time Gesture Recognition Demo
====================================
Live webcam demo with gesture recognition in real-time.

Controls:
    q        → Quit
    s        → Screenshot to models/
    r        → Reset temporal smoothing
    f        → Toggle FPS display
    h        → Toggle help overlay

Usage:
    python demo/demo.py
    python demo/demo.py --onnx          # ONNX Runtime backend (faster)
    python demo/demo.py --camera 1      # Use camera index 1
    python demo/demo.py --camera 0 --width 1280 --height 720
"""

import cv2
import numpy as np
import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.pipeline import GestureRecognitionPipeline
from gestures.utils import draw_text_with_bg, draw_confidence_bar

# ── Gesture color palette ──
GESTURE_COLORS = {
    "point_left":   (255, 100,   0),
    "point_right":  (  0, 200, 255),
    "point_up":     (100, 255, 100),
    "point_down":   (255, 100, 255),
    "open_palm":    (  0, 255,   0),
    "thumbs_up":    (  0, 255, 200),
    "thumbs_down":  (  0,   0, 255),
    "pinch":        (255, 255,   0),
    "wave":         (200, 150, 255),
}


def get_camera_backends():
    """Return the backend probes to try for this platform."""
    if sys.platform == "win32":
        return [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]
    return [("ANY", cv2.CAP_ANY)]


def show_camera_error_window(message: str, width: int, height: int):
    """Show a visible error window when the demo is launched without a console."""
    frame = np.zeros((max(height, 420), max(width, 900), 3), dtype=np.uint8)
    lines = ["Camera startup failed."] + message.splitlines() + ["", "Press Q or Esc to close."]

    for i, line in enumerate(lines):
        color = (220, 220, 220)
        if i == 0:
            color = (0, 80, 255)
        draw_text_with_bg(
            frame,
            line,
            (30, 50 + i * 32),
            font_scale=0.65 if i == 0 else 0.52,
            color=color,
            bg_color=(20, 20, 20),
        )

    window_name = "Gesture Recognition Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(50) & 0xFF
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if key in (ord("q"), 27) or visible < 1:
            break
    cv2.destroyAllWindows()


def open_camera_with_fallback(preferred_index: int, width: int, height: int, max_index: int = 10):
    """
    Try to open a camera. If preferred_index fails, scan additional indices.
    Returns (cap, actual_index) or raises RuntimeError if all fail.
    """
    indices_to_try = [preferred_index] + [i for i in range(max_index + 1) if i != preferred_index]
    attempted = []

    for idx in indices_to_try:
        print(f"  Trying camera index {idx}...", end=" ", flush=True)

        for backend_name, backend in get_camera_backends():
            attempted.append(f"{idx}:{backend_name}")
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Warm up: try to read 3 frames
            success_count = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    success_count += 1
                time.sleep(0.05)
            
            if success_count >= 2:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ ({backend_name}, camera {idx}, {actual_w}×{actual_h})")
                return cap, idx
            
            cap.release()
        
        print("✗")
    
    attempted_str = ", ".join(attempted)
    raise RuntimeError(
        f"No working camera found after scanning indices 0-{max_index}.\n"
        f"  Tried: {attempted_str}\n"
        "  - Close any app already using the webcam.\n"
        "  - On Windows, enable Camera access and 'Let desktop apps access your camera'.\n"
        "  - Try running Camera app first to confirm the webcam works.\n"
        "  - Re-run with a specific index: python demo/demo.py --camera 1"
    )


def read_frame_robust(cap, max_retries: int = 10):
    """
    Read a frame with retry logic. Returns (True, frame) or (False, None).
    """
    for _ in range(max_retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(0.01)
    return False, None


def main():
    """Run the real-time gesture recognition demo."""
    parser = argparse.ArgumentParser(description="Real-time gesture recognition demo")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Preferred camera index (will fall back to others if unavailable)")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX Runtime backend (faster)")
    parser.add_argument("--model", type=str, default=None, help="Override model file path")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal flip (mirror mode is on by default)")
    args = parser.parse_args()

    # ── Banner ──
    print("=" * 60)
    print("  Real-Time Gesture Recognition System")
    print("=" * 60)

    # ── Initialize components ──
    print("\n[1/3] Loading end-to-end pipeline...")
    use_onnx = args.onnx
    try:
        pipeline = GestureRecognitionPipeline(
            model_path=args.model,
            use_onnx=use_onnx,
        )
        backend_label = "ONNX" if use_onnx else "PyTorch"
        print(f"      ✓ Pipeline ready ({backend_label} backend)")
    except Exception as e:
        if use_onnx:
            print(f"      ✗ ONNX pipeline load failed ({e}), falling back to PyTorch")
            use_onnx = False
            pipeline = GestureRecognitionPipeline(
                model_path=args.model,
                use_onnx=False,
            )
            print("      ✓ Pipeline ready (PyTorch backend)")
        else:
            raise

    # ── Open camera ──
    print(f"\n[2/3] Opening camera (preferred: {args.camera})...")
    try:
        cap, actual_camera_idx = open_camera_with_fallback(args.camera, args.width, args.height)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        pipeline.release()
        show_camera_error_window(str(e), args.width, args.height)
        return

    print("[3/3] Starting render loop...")
    print("\n" + "=" * 60)
    print("  Demo running! Controls: q=Quit  s=Screenshot  r=Reset  h=Help")
    print("=" * 60 + "\n")

    show_fps = True
    show_help = False
    screenshot_count = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30  # About 1 second at 30fps before warning

    try:
        # ── Main loop ──
        while True:
            # Robust frame read
            ret, frame = read_frame_robust(cap, max_retries=3)
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("\nWARNING: Camera is not producing frames.")
                    print("  Options:")
                    print("    1. Press 'q' to quit and try --camera 1")
                    print("    2. Reconnect your webcam")
                    consecutive_failures = 0  # Reset and keep trying
                placeholder = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                draw_text_with_bg(placeholder, "Camera not available — press 'q' to quit",
                                  (50, placeholder.shape[0] // 2), font_scale=0.6, color=(0, 80, 255))
                cv2.imshow("Gesture Recognition Demo", placeholder)
                if (cv2.waitKey(100) & 0xFF) == ord("q"):
                    break
                continue

            consecutive_failures = 0

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            results = pipeline.process_frame(frame)

            for result in results:
                try:
                    pipeline.detector.draw_landmarks_data(
                        frame,
                        result.landmarks,
                        result.bbox,
                        draw_bbox=(result.gesture is not None),
                    )

                    if result.gesture is not None:
                        color = GESTURE_COLORS.get(result.gesture, (255, 255, 255))
                        x1, y1, x2, y2 = result.bbox
                        label = f"{result.gesture.replace('_', ' ').upper()} ({result.confidence:.0%})"
                        draw_text_with_bg(
                            frame, label, (x1, max(0, y1 - 12)),
                            font_scale=0.65, color=color, bg_color=(20, 20, 20),
                        )
                        draw_text_with_bg(
                            frame, f"{result.handedness} hand",
                            (x1, y2 + 22), font_scale=0.45, color=(200, 200, 200),
                        )
                        draw_confidence_bar(
                            frame, result.gesture, result.confidence,
                            position=(x1, y2 + 40), bar_width=max(x2 - x1, 60),
                        )

                        top3_idx = np.argsort(result.all_probabilities)[-3:][::-1]
                        for rank, idx in enumerate(top3_idx):
                            g_name = config.IDX_TO_GESTURE[idx]
                            g_prob = result.all_probabilities[idx]
                            draw_text_with_bg(
                                frame, f"  {g_name}: {g_prob:.2f}",
                                (x1, y2 + 68 + rank * 18),
                                font_scale=0.38, color=(170, 170, 170),
                            )
                    else:
                        x1, y1, _, _ = result.bbox
                        draw_text_with_bg(
                            frame, "detecting...",
                            (x1, max(0, y1 - 10)),
                            font_scale=0.5, color=(100, 100, 255),
                        )

                except Exception:
                    continue

            if show_fps:
                fps = pipeline.get_fps()
                latency_stats = pipeline.get_latency_stats()
                latency_ms = latency_stats.get("mean", 0.0)
                fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
                hud_lines = [
                    (f"FPS: {fps:.1f}", fps_color, 0.7),
                    (f"Mean latency: {latency_ms:.1f} ms", (200, 200, 200), 0.5),
                    (f"Hands: {len(results)}", (200, 200, 200), 0.5),
                    (f"Backend: {backend_label}", (200, 200, 200), 0.5),
                    (f"Camera: {actual_camera_idx}", (200, 200, 200), 0.5),
                ]
                for i, (text, color, scale) in enumerate(hud_lines):
                    draw_text_with_bg(frame, text, (10, 30 + i * 25), font_scale=scale, color=color)

            if show_help:
                help_lines = ["q → Quit", "s → Screenshot", "r → Reset", "f → FPS", "h → Help"]
                for i, line in enumerate(help_lines):
                    draw_text_with_bg(
                        frame, line, (frame.shape[1] - 185, 30 + i * 25),
                        font_scale=0.45, color=(255, 220, 80),
                    )

            cv2.imshow("Gesture Recognition Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                config.MODELS_DIR.mkdir(exist_ok=True)
                path = config.MODELS_DIR / f"screenshot_{screenshot_count}.png"
                cv2.imwrite(str(path), frame)
                screenshot_count += 1
                print(f"Screenshot saved: {path}")
            elif key == ord("r"):
                pipeline.reset()
                print("Temporal smoothing state reset")
            elif key == ord("f"):
                show_fps = not show_fps
            elif key == ord("h"):
                show_help = not show_help
    except KeyboardInterrupt:
        print("\nStopping demo...")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    pipeline.release()

    # Print session stats
    stats = pipeline.get_latency_stats()
    if stats.get("mean", 0) > 0:
        metrics = {
            "backend": backend_label,
            "camera_index": actual_camera_idx,
            "mean_fps": pipeline.get_fps(),
            "latency_ms": stats,
            "frame_size": {
                "width": args.width,
                "height": args.height,
            },
        }
        metrics_path = config.MODELS_DIR / "runtime_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("  Session Statistics")
        print(f"  Backend      : {backend_label}")
        print(f"  Camera index : {actual_camera_idx}")
        print(f"  Mean latency : {stats['mean']:.1f} ms")
        print(f"  P50 latency  : {stats['p50']:.1f} ms")
        print(f"  P95 latency  : {stats['p95']:.1f} ms")
        print(f"  Min / Max    : {stats['min']:.1f} / {stats['max']:.1f} ms")
        print(f"  Mean FPS     : {pipeline.get_fps():.1f}")
        print(f"  Metrics file : {metrics_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
