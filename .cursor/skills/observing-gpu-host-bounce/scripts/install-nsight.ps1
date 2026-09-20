# Add Nsight Systems, Nsight Compute, and Compute Sanitizer to the existing
# CUDA 12.6 toolkit. Do not install Nvidia.CUDA 13.x.
param(
    [string]$Installer = "C:\Users\johnh\TRM\.cache\cuda_12.6.3_windows_network.exe"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Installer)) {
    throw "Installer missing: $Installer. Download cuda_12.6.3_windows_network.exe first."
}

Write-Host "Installing nsight_systems_12.6 nsight_compute_12.6 sanitizer_12.6 from $Installer"
$args = @("-s", "nsight_systems_12.6", "nsight_compute_12.6", "sanitizer_12.6")
$p = Start-Process -FilePath $Installer -ArgumentList $args -Wait -PassThru
if ($p.ExitCode -ne 0) {
    throw "CUDA Nsight installer exited $($p.ExitCode)"
}

$nsys = Get-ChildItem "${env:ProgramFiles}\NVIDIA Corporation", "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.6" -Recurse -Filter nsys.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$ncu = Get-ChildItem "${env:ProgramFiles}\NVIDIA Corporation", "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.6" -Recurse -Filter ncu.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$san = Get-ChildItem "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.6" -Recurse -Filter compute-sanitizer.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
Write-Host "nsys              : $(if ($nsys) { $nsys } else { 'MISSING' })"
Write-Host "ncu               : $(if ($ncu) { $ncu } else { 'MISSING' })"
Write-Host "compute-sanitizer : $(if ($san) { $san } else { 'MISSING' })"
if (-not $nsys -or -not $ncu) {
    throw "Install finished but nsys/ncu still missing. Re-run elevated."
}
