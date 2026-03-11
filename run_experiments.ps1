$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SEEDS = @(100, 200, 300, 400, 500)
$TOTAL_STEPS = 1000000
$DEVICE = "auto"
$env:PYTHONUNBUFFERED = "1"
$PythonExe = "D:\Work\Miniconda\envs\default\python.exe"
$EVAL_SEEDS = @(0, 42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384)
$EPISODES_PER_SEED = 50

function Write-Status {
    param(
        [string]$Message,
        [string]$Color = "Gray"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Format-Duration {
    param(
        [TimeSpan]$Elapsed
    )

    return "{0:00}:{1:00}:{2:00}" -f [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds
}

function Assert-Artifact {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path $Path)) {
        throw "$Label was not created: $Path"
    }

    Write-Status "Verified $Label -> $Path" "Green"
}

function Invoke-Python {
    param(
        [string[]]$ArgsList,
        [string]$Label = "",
        [string[]]$ExpectedPaths = @()
    )

    $startedAt = Get-Date
    if ($Label) {
        Write-Status "$Label started." "Yellow"
    }

    Write-Host ($PythonExe + " " + ($ArgsList -join " "))
    & $PythonExe @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }

    foreach ($path in $ExpectedPaths) {
        Assert-Artifact -Path $path -Label $path
    }

    if ($Label) {
        $elapsed = (Get-Date) - $startedAt
        Write-Status "$Label finished in $(Format-Duration $elapsed)." "Green"
    }
}

Write-Host "=========================================================="
Write-Host "Starting Multi-Seed Curriculum Pipeline (AeroPursuit)"
Write-Host "Hardware target: RTX 5080"
Write-Host "=========================================================="
if (-not (Test-Path $PythonExe)) {
    throw "Conda default python not found: $PythonExe"
}
Write-Status "Runtime env: conda default; device: $DEVICE; Python output is unbuffered." "Cyan"
Write-Status "Stage milestones and artifact checks will be printed below." "Cyan"

$curriculumPairs = New-Object System.Collections.Generic.List[string]

for ($seedIndex = 0; $seedIndex -lt $SEEDS.Count; $seedIndex++) {
    $seed = $SEEDS[$seedIndex]
    Write-Host "`n>>> Starting SEED $seed ($($seedIndex + 1)/$($SEEDS.Count)) <<<" -ForegroundColor Cyan

    $currDir = "results/curriculum_seed_$seed"

    Write-Status "[Seed $seed] Curriculum Stage 1 / 2: training hen." "Yellow"
    Invoke-Python -ArgsList @(
        "src/train_hen.py",
        "--seed", [string]$seed,
        "--total-steps", [string]$TOTAL_STEPS,
        "--save-dir", $currDir,
        "--device", $DEVICE
    ) -Label "[Seed $seed] train_hen.py" -ExpectedPaths @(
        (Join-Path $currDir "hen_stage_1.zip"),
        (Join-Path $currDir "best_hen\best_model.zip")
    )

    Write-Status "[Seed $seed] Curriculum Stage 2 / 2: training eagle." "Yellow"
    $henModelPath = Join-Path $currDir "best_hen/best_model.zip"
    $eagleModelPath = Join-Path $currDir "best_eagle/best_model.zip"
    Invoke-Python -ArgsList @(
        "src/train_eagle.py",
        "--seed", [string]$seed,
        "--total-steps", [string]$TOTAL_STEPS,
        "--save-dir", $currDir,
        "--hen-model", $henModelPath,
        "--device", $DEVICE
    ) -Label "[Seed $seed] train_eagle.py" -ExpectedPaths @(
        (Join-Path $currDir "eagle_stage_1.zip"),
        $eagleModelPath
    )

    $curriculumPairs.Add("${henModelPath}::${eagleModelPath}")
    Write-Status "[Seed $seed] Curriculum training completed." "Cyan"
}

Write-Host "`n=========================================================="
Write-Host "All curriculum runs completed. Running final evaluation..."
Write-Host "=========================================================="

$finalEvalArgs = @(
    "src/evaluate.py",
    "--method", "all",
    "--eval-seeds"
)

foreach ($evalSeed in $EVAL_SEEDS) {
    $finalEvalArgs += [string]$evalSeed
}

$finalEvalArgs += @(
    "--episodes-per-seed", [string]$EPISODES_PER_SEED,
    "--device", $DEVICE
)

foreach ($pair in $curriculumPairs) {
    $finalEvalArgs += @("--curriculum-pair", $pair)
}

Invoke-Python -ArgsList $finalEvalArgs -Label "Final evaluation" -ExpectedPaths @(
    "results/eval_episode.csv",
    "results/eval_seed_summary.csv",
    "results/eval_run_summary.csv",
    "AeroPursuit-Predator-Prey-for-EERC2026/generated/latex_tables.tex"
)

Write-Host "=========================================================="
Write-Host "Experiments finished."
Write-Host "Check results/eval_run_summary.csv and AeroPursuit-Predator-Prey-for-EERC2026/generated/latex_tables.tex."
Write-Host "=========================================================="
