"""Setup script for the gesture recognition package."""

from setuptools import setup, find_packages

setup(
    name="gesture-recognition",
    version="1.0.0",
    description="Real-time hand gesture recognition system using MediaPipe and PyTorch",
    author="Gesture Recognition Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "mediapipe>=0.10.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
    ],
    entry_points={
        "console_scripts": [
            "gesture-demo=demo.demo:main",
        ],
    },
)
