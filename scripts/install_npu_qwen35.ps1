<#
.SYNOPSIS
    Install Qwen3.5-4B VLM for Intel NPU on Windows 11 (Core Ultra 155H).
    Pulls model from Hugging Face, exports to ONNX, quantizes to INT4 with NNCF,
    and builds OpenVINO IR ready for OpenVINO GenAI.

.NOTES
    Run as Administrator in PowerShell 7+.
    Requires: Intel NPU driver 16.x+, Arc iGPU driver 20.x+, Python 3.11+, Git, wget/curl.

.EXAMPLE
    .\scripts\install_npu_qwen35.ps1 -ModelDir "C:\Models\qwen35-4b-int4-ov" -PythonPath "C:\Python311\python.exe"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelDir,

    [Parameter(Mandatory=$false)]
    [string]$PythonPath = "python",

    [Parameter(Mandatory=$false)]
    [string]$RepoId = "Qwen/Qwen3.5-4B",

    [Parameter(Mandatory=$false)]
    [string]$Quantization = "int4",   # int4 | int8

    [Parameter(Mandatory=$false)]
    [int]$ContextLength = 4096
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $Message"
}

# ---------------------------------------------------------------------------
# 1. Verify drivers & Python
# ---------------------------------------------------------------------------
Write-Log "Checking environment..."
$pyVer = & $PythonPath --version 2>$null
if (-not $?) { Write-Error "Python not found at $PythonPath"; exit 1 }
Write-Log "Python: $pyVer"

# Check Intel NPU plugin
$ovPlugin = & $PythonPath -c "import openvino; print('openvino', openvino.__version__); core = openvino.Core(); print('devices:', core.available_devices)" 2>$null
if (-not $?) { Write-Error "OpenVINO not importable. Run: $PythonPath -m pip install openvino==2026.4.1"; exit 1 }
Write-Log "OpenVINO available devices: $ovPlugin"

# Ensure NPU device is present
if ($ovPlugin -notmatch "NPU") {
    Write-Warning "NPU device not detected. Verify Intel NPU driver is installed (Intel Driver & Support Assistant)."
}

# ---------------------------------------------------------------------------
# 2. Install Python dependencies
# ---------------------------------------------------------------------------
Write-Log "Installing Python packages..."
$pipArgs = @(
    "openvino==2026.4.1"
    "openvino-genai"
    "openvino-intel-npu-plugin"
    "nncf"
    "torch"
    "transformers"
    "optimum[exporters]"
    "huggingface_hub"
    "safetensors"
    "pillow"
    "dxcam"
    "pynput"
    "pywin32"
)
foreach ($pkg in $pipArgs) {
    Write-Log "Installing $pkg..."
    & $PythonPath -m pip install $pkg
    if (-not $?) { Write-Error "Failed to install $pkg"; exit 1 }
}

# ---------------------------------------------------------------------------
# 3. Prepare model directory
# ---------------------------------------------------------------------------
$modelDir = Resolve-Path $ModelDir
if (-not (Test-Path $modelDir)) {
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
}
Write-Log "Model directory: $modelDir"

# ---------------------------------------------------------------------------
# 4. Export to ONNX via Optimum
# ---------------------------------------------------------------------------
$onnxDir = Join-Path $modelDir "onnx"
if (-not (Test-Path $onnxDir)) { New-Item -ItemType Directory -Path $onnxDir -Force | Out-Null }

Write-Log "Exporting $RepoId to ONNX ($onnxDir)..."
& $PythonPath -m optimum.exporters.onnx --model $RepoId --task image-text-to-text --output $onnxDir --opset 17 --trust-remote-code
if (-not $?) { Write-Error "ONNX export failed"; exit 1 }

# ---------------------------------------------------------------------------
# 5. Quantize with NNCF to INT4 / INT8
# ---------------------------------------------------------------------------
$irDir = Join-Path $modelDir "int4-ov"
if ($Quantization -eq "int8") { $irDir = Join-Path $modelDir "int8-ov" }
if (-not (Test-Path $irDir)) { New-Item -ItemType Directory -Path $irDir -Force | Out-Null }

Write-Log "Quantizing to $Quantization via NNCF (output: $irDir)..."

$quantScript = @"
import nncf
import openvino as ov
from pathlib import Path

onnx_model = Path(r"$onnxDir")
ir_model = Path(r"$irDir")
quant = "$Quantization"

core = ov.Core()
model = core.read_model(str(onnx_model))

if quant == "int4":
    # INT4 weight-only quantization via MatMulNBits (4-bit)
    compressed_model = nncf.quantize(
        model,
        mode=nncf.QuantizationMode.INT4,
        target_device=nncf.TargetDevice.NPU,
        subset_size=128,  # calibration dataset size
        ignored_scope=nncf.IgnoredScope(
            patterns=["*/vision*", "*/patch_embed*", "*/positional*"]
        ),
    )
else:
    # INT8 symmetric quantization
    compressed_model = nncf.quantize(
        model,
        mode=nncf.QuantizationMode.INT8,
        target_device=nncf.TargetDevice.NPU,
        subset_size=128,
        ignored_scope=nncf.IgnoredScope(
            patterns=["*/vision*", "*/patch_embed*", "*/positional*"]
        ),
    )

ov.save_model(compressed_model, str(ir_model / "model.xml"))
print(f"Saved quantized IR to {ir_model}")
"@

$quantPath = Join-Path $env:TEMP "quantize_qwen35.py"
$quantScript | Set-Content -Path $quantPath -Encoding UTF8
& $PythonPath $quantPath
if (-not $?) { Write-Error "NNCF quantization failed"; exit 1 }

# ---------------------------------------------------------------------------
# 6. Verify IR loads on NPU
# ---------------------------------------------------------------------------
Write-Log "Verifying IR on NPU..."
$verifyScript = @"
import openvino as ov
core = ov.Core()
model = core.read_model(r"$irDir\model.xml")
compiled = core.compile_model(model, "NPU")
print("NPU compilation successful")
print("Input shapes:", [i.shape for i in compiled.inputs])
print("Output shapes:", [o.shape for o in compiled.outputs])
"@
$verifyPath = Join-Path $env:TEMP "verify_ir.py"
$verifyScript | Set-Content -Path $verifyPath -Encoding UTF8
& $PythonPath $verifyPath
if (-not $?) { Write-Error "IR verification on NPU failed"; exit 1 }

# ---------------------------------------------------------------------------
# 7. Print summary
# ---------------------------------------------------------------------------
Write-Log "============================================================"
Write-Log "Installation complete."
Write-Log "Quantized OpenVINO IR: $irDir"
Write-Log "Model repo: $RepoId"
Write-Log "Quantization: $Quantization"
Write-Log "Context length: $ContextLength"
Write-Log ""
Write-Log "Next steps:"
Write-Log "  1. Copy the IR directory to your daemon config:"
Write-Log "     model.ov_model_dir: $irDir"
Write-Log "  2. Run the daemon (see run_daemon.ps1):"
Write-Log "     .\scripts\run_daemon.ps1 -ModelDir $irDir"
Write-Log "============================================================"