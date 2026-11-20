<#
.SYNOPSIS
    Run the NPU screen-reading daemon (continuous capture → inference → OSD/hotkey).
.DESCRIPTION
    Starts the Python daemon that uses the npu_screen package.  The script
    assumes the model IR has already been built (see install_npu_qwen35.ps1).

.PARAMETER ModelDir
    Path to the OpenVINO IR directory containing model.xml and model.bin.
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$ModelDir
)

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
Write-Host "Checking Python..."
python --version
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found"; exit 1 }

# ---------------------------------------------------------------------------
# Build Python command line
# ---------------------------------------------------------------------------
$pythonExe = "python"
# Ensure model dir is in PYTHONPATH
$env:PYTHONPATH = (Get-Location) + ";$env:PYTHONPATH"

# Pass model IR path via environment variable for AppConfig to read
$env:MODEL_IR = $ModelDir
Write-Host "Model IR path: $env:MODEL_IR"

# ---------------------------------------------------------------------------
# Start daemon
# ---------------------------------------------------------------------------
Write-Host "Launching NPU screen daemon..."
& $pythonExe -m thomasllm.npu_screen.cli --config thomasllm/npu_screen/configs/default.yaml
if ($LASTEXITCODE -ne 0) { Write-Error "Daemon failed to start"; exit 1 }