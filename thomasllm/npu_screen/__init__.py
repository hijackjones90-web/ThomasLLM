"""thomasllm.npu_screen - Qwen3.5 VLM on Intel NPU with real-time screen reading.

Target: Windows 11, Intel Core Ultra 155H (NPU + Arc iGPU), 16 GB RAM.
Runs Qwen3.5-4B VLM (INT4) exclusively on the Intel NPU via OpenVINO GenAI,
with continuous dxcam screen capture and OSD/hotkey/pipe output sinks.
"""

__version__ = "0.1.0"