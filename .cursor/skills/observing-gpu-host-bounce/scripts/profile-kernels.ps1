# Find nsys / ncu / compute-sanitizer on this Windows box and run the
# matching TRM-Omega packed-inference profile.
param(
    [ValidateSet("nsys", "ncu", "sanitizer", "which")]
    [string]$Mode = "which",
    [string]$Kernel = "fused_attention",
    [string]$Crate = ""
)

$ErrorActionPreference = "Stop"

function Find-OnPath([string]$exe) {
    $cmd = Get-Command $exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Find-NsightSystemsNsys {
    $root = Join-Path ${env:ProgramFiles} "NVIDIA Corporation"
    if (-not (Test-Path $root)) { return $null }
    $hit = Get-ChildItem $root -Directory -Filter "Nsight Systems *" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object {
            $p = Join-Path $_.FullName "target-windows-x64\nsys.exe"
            if (Test-Path $p) { $p }
        } |
        Select-Object -First 1
    return $hit
}

function Find-Tool([string]$exe) {
    $pathHit = Find-OnPath $exe
    if ($pathHit) { return $pathHit }
    if ($exe -eq "nsys.exe") {
        $sys = Find-NsightSystemsNsys
        if ($sys) { return $sys }
    }
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

function Get-ServerExe {
    Write-Host "building server (release, cuda)..."
    cargo build --release --features cuda --bin server
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed: $LASTEXITCODE" }
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Crate "target" }
    $exe = Join-Path $target "release\server.exe"
    if (-not (Test-Path $exe)) { throw "server.exe missing at $exe" }
    return $exe
}

$serverArgs = @("--kernels", "--batch", "1", "--seq-len", "81", "--iters", "2", "--n-l", "2", "--n-sup", "2")

switch ($Mode) {
    "nsys" {
        if (-not $nsys) { throw "nsys.exe not installed. Run install-nsight.ps1" }
        $exe = Get-ServerExe
        $rep = Join-Path $art "kernels.nsys-rep"
        # Nsight Compute's bundled nsys 2024.3 needs admin ETW; prefer Systems 2024.5+.
        # Profile the binary, not cargo. --sample/--cpuctxsw none avoids admin CPU tracing.
        $ErrorActionPreference = "Continue"
        & $nsys profile -f true -t cuda -s none --cpuctxsw=none -o $rep $exe @serverArgs
        if ($LASTEXITCODE -ne 0) { throw "nsys profile failed: $LASTEXITCODE" }
        & $nsys stats --force-export=true --report cuda_gpu_kern_sum --report cuda_api_sum $rep
    }
    "ncu" {
        if (-not $ncu) { throw "ncu.exe not installed. Run install-nsight.ps1" }
        $exe = Get-ServerExe
        $rep = Join-Path $art "kernels.ncu-rep"
        $ErrorActionPreference = "Continue"
        & $ncu --force-overwrite --kernel-name $Kernel --set full -o $rep $exe @serverArgs
    }
    "sanitizer" {
        if (-not $san) { throw "compute-sanitizer.exe not installed. Run install-nsight.ps1" }
        & $san --tool memcheck cargo test --features cuda --test kernel_parity -- --test-threads=1 cuda_kernelbank_matches_cpu_ref
    }
}
