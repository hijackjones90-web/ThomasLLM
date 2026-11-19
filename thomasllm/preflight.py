#!/usr/bin/env python3
"""Preflight checks for ThomasVLM training.

Run this before any download or training. Exits with code 2 if the machine
cannot support the planned run, and writes a report to --out.
"""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "peft",
    "safetensors",
    "PIL",
    "numpy",
    "matplotlib",
    "huggingface_hub",
    "sentencepiece",
    "accelerate",
]

OPTIONAL_MODULES = ["bitsandbytes", "sacrebleu", "cv2"]

MIN_VRAM_GB = 20.0
MIN_FREE_DISK_GB = 25.0


def sh(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )
    except Exception as exc:
        return subprocess.CompletedProcess(cmd, 1, "", str(exc))


def module_status(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"present": False, "version": None}
    try:
        mod = importlib.import_module(name)
        return {"present": True, "version": getattr(mod, "__version__", "?")}
    except Exception as exc:
        return {"present": True, "version": None, "import_error": str(exc)}


def collect():
    report = {
        "timestamp": None,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "cpu_count": os.cpu_count(),
        "disk": {},
        "gpu": {},
        "modules": {},
        "network": {},
    }
    import datetime

    report["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

    # Disk
    for path in ["/workspace", "/tmp", "/artifacts"]:
        if os.path.isdir(path):
            usage = shutil.disk_usage(path)
            report["disk"][path] = {
                "total_gb": round(usage.total / 1e9, 2),
                "free_gb": round(usage.free / 1e9, 2),
                "used_gb": round(usage.used / 1e9, 2),
            }

    # GPU via nvidia-smi
    smi = sh("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version "
             "--format=csv,noheader")
    if smi.returncode == 0 and smi.stdout.strip():
        entries = []
        for line in smi.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                entries.append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "memory_used": parts[2],
                        "driver": parts[3],
                    }
                )
        report["gpu"]["nvidia_smi"] = entries

    # Torch view of CUDA
    torch_info = module_status("torch")
    report["modules"]["torch"] = torch_info
    if torch_info["present"] and not torch_info.get("import_error"):
        import torch

        cuda_available = bool(torch.cuda.is_available())
        report["gpu"]["cuda_available"] = cuda_available
        report["gpu"]["torch_cuda"] = torch.version.cuda
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            report["gpu"]["device_name"] = props.name
            report["gpu"]["vram_gb"] = round(props.total_memory / 1e9, 2)
            report["gpu"]["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        else:
            report["gpu"]["device_name"] = None
            report["gpu"]["vram_gb"] = 0.0
            report["gpu"]["bf16_supported"] = False

    for name in REQUIRED_MODULES + OPTIONAL_MODULES:
        if name == "torch":
            continue
        report["modules"][name] = module_status(name)

    # Network
    hf = sh("curl -sI --max-time 20 https://huggingface.co")
    report["network"]["huggingface"] = {
        "reachable": hf.returncode == 0 and "200" in (hf.stdout or ""),
        "status": (hf.stdout or "").splitlines()[0] if hf.stdout else "",
    }
    probe = sh(
        "curl -sI --max-time 20 -o /dev/null -w '%{http_code}' "
        "https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/config.json"
    )
    report["network"]["hf_model_probe"] = probe.stdout.strip()

    return report


def verdict(report):
    problems = []
    gpu = report.get("gpu", {})
    vram = gpu.get("vram_gb", 0.0) or 0.0
    if not gpu.get("cuda_available", False):
        problems.append("No CUDA device visible to torch")
    if vram < MIN_VRAM_GB:
        problems.append(
            f"VRAM {vram:.1f} GB < required {MIN_VRAM_GB:.0f} GB "
            "(1B bf16 VLM with frozen vision tower)"
        )
    free_gb = report.get("disk", {}).get("/workspace", {}).get("free_gb", 0.0)
    if free_gb and free_gb < MIN_FREE_DISK_GB:
        problems.append(
            f"Free disk {free_gb:.1f} GB < required {MIN_FREE_DISK_GB:.0f} GB"
        )
    missing = [
        n
        for n in REQUIRED_MODULES
        if not report["modules"].get(n, {}).get("present")
    ]
    if missing:
        problems.append("Missing modules: " + ", ".join(missing))
    if not report.get("network", {}).get("huggingface", {}).get("reachable"):
        problems.append("huggingface.co not reachable")

    report["verdict"] = {
        "feasible": not problems,
        "blocking_problems": problems,
        "mode": "full-training" if not problems else "feasibility-only",
    }
    return report


def print_summary(report):
    gpu = report.get("gpu", {})
    print("python:", report["python"]["version"].split()[0])
    print("cpu:", report["cpu_count"])
    for path, d in report.get("disk", {}).items():
        print(f"disk {path}: {d['free_gb']:.1f} GB free / {d['total_gb']:.1f} GB")
    print("cuda:", gpu.get("cuda_available"))
    print("gpu:", gpu.get("device_name"), gpu.get("vram_gb"), "GB")
    print("bf16:", gpu.get("bf16_supported"))
    missing = [
        n for n in REQUIRED_MODULES if not report["modules"].get(n, {}).get("present")
    ]
    print("missing modules:", missing or "none")
    print("hf reachable:", report.get("network", {}).get("huggingface", {}).get("reachable"))
    print()
    v = report["verdict"]
    print("feasible:", v["feasible"], "| mode:", v["mode"])
    for p in v["blocking_problems"]:
        print("  -", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/artifacts/thomasvlm/preflight.json")
    ap.add_argument("--txt", default="/artifacts/thomasvlm/preflight.txt")
    args = ap.parse_args()

    report = verdict(collect())
    print_summary(report)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    lines = [f"{k}: {v}" for k, v in report["verdict"].items() if k != "blocking_problems"]
    lines.append("blocking_problems:")
    lines += ["  - " + p for p in report["verdict"]["blocking_problems"]]
    Path(args.txt).write_text(
        "ThomasVLM preflight\n"
        + json.dumps(report, indent=2)
        + "\n\nverdict\n"
        + "\n".join(lines)
        + "\n"
    )
    sys.exit(0 if report["verdict"]["feasible"] else 2)


if __name__ == "__main__":
    main()