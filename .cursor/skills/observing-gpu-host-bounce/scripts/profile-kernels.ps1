# Find nsys / ncu / compute-sanitizer on this Windows box and run the
# matching TRM-Omega packed-inference profile. Default crate is the nested
# historical folder name.
param(
    [ValidateSet("nsys", "ncu", "sanitizer", "which")]
    [string]$Mode = "which",
    [string]$Kernel = "fused_attention",
    [string]$Crate = ""
)

$ErrorActionPreference = "Stop"

function Find-Tool([string]$exe) {
    $cmd = Get-Command $exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $roots = @(
        "${env:ProgramFiles}\NVIDIA Corporation",
        "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $hit = Get-ChildItem $root -Recurse -Filter $exe -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
        if ($hit) { return $hit }
    }
    return $null
}

$nsys = Find-Tool "nsys.exe"
$ncu = Find-Tool "ncu.exe"
$san = Find-Tool "compute-sanitizer.exe"

if (-not $Crate) {
    $root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
    foreach ($name in @("trm_omega", "trm-deq-convergence-&-neumann-verification")) {
        $candidate = Join-Path $root $name
        if (Test-Path (Join-Path $candidate "Cargo.toml")) {
            $Crate = $candidate
            break
        }
    }
    if (-not $Crate) {
        throw "Could not find crate (looked for trm_omega under $root)"
    }
}

Write-Host "nsys              : $(if ($nsys) { $nsys } else { 'MISSING' })"
Write-Host "ncu               : $(if ($ncu) { $ncu } else { 'MISSING' })"
Write-Host "compute-sanitizer : $(if ($san) { $san } else { 'MISSING' })"
Write-Host "crate             : $Crate"

if ($Mode -eq "which") { exit 0 }

Set-Location $Crate
$art = Join-Path $Crate "artifacts"
New-Item -ItemType Directory -Force -Path $art | Out-Null

$server = @("run", "--release", "--features", "cuda", "--bin", "server", "--", "--kernels", "--batch", "1", "--seq-len", "81", "--iters", "2", "--n-l", "2", "--n-sup", "2")

switch ($Mode) {
    "nsys" {
        if (-not $nsys) { throw "nsys.exe not installed. Run install-nsight.ps1" }
        $rep = Join-Path $art "kernels.nsys-rep"
        & $nsys profile --force-overwrite true --trace=cuda,nvtx,osrt --output $rep -- cargo @server
        & $nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum $rep
    }
    "ncu" {
        if (-not $ncu) { throw "ncu.exe not installed. Run install-nsight.ps1" }
        $rep = Join-Path $art "kernels.ncu-rep"
        & $ncu --force-overwrite --target-processes all --kernel-name $Kernel --set full -o $rep cargo @server
    }
    "sanitizer" {
        if (-not $san) { throw "compute-sanitizer.exe not installed. Run install-nsight.ps1" }
        & $san --tool memcheck cargo test --features cuda --test kernel_parity -- --test-threads=1 cuda_kernelbank_matches_cpu_ref
    }
}
