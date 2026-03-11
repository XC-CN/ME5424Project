$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SEEDS = @(100, 200, 300, 400, 500)
$TOTAL_STEPS = 1000000
$CO_ROUNDS = 20
$CO_STEPS_PER_ROUND = 50000
$DEVICE = "auto"

# Use a small held-out selection set to choose the best co-training round per seed.
$MODEL_SELECT_SEEDS = @(9001, 9002)
$MODEL_SELECT_EPISODES = 10

function Invoke-Python {
    param(
        [string[]]$ArgsList
    )
    Write-Host ("python " + ($ArgsList -join " "))
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Select-BestCoTrainingPair {
    param(
        [string]$CoDir,
        [int]$Seed,
        [int]$Rounds
    )

    $bestPair = $null
    $bestRound = $null
    $bestCaptureRate = [double]::PositiveInfinity
    $bestAvgTailDist = [double]::NegativeInfinity
    $bestEpisodeLength = [double]::NegativeInfinity

    for ($round = 1; $round -le $Rounds; $round++) {
        $henRoundPath = Join-Path $CoDir ("hen_round_{0}.zip" -f $round)
        $eagleRoundPath = Join-Path $CoDir ("eagle_round_{0}.zip" -f $round)

        if (-not (Test-Path $henRoundPath) -or -not (Test-Path $eagleRoundPath)) {
            throw "Missing co-training checkpoint for seed $Seed round $round"
        }

        $tmpPrefix = Join-Path "results" ("tmp_select_seed_{0}_round_{1}" -f $Seed, $round)
        $episodeCsv = "${tmpPrefix}_episode.csv"
        $seedCsv = "${tmpPrefix}_seed.csv"
        $runCsv = "${tmpPrefix}_run.csv"
        $latexTex = "${tmpPrefix}_tables.tex"

        $argsList = @(
            "src/evaluate.py",
            "--method", "cotraining",
            "--eval-seeds"
        )
        foreach ($evalSeed in $MODEL_SELECT_SEEDS) {
            $argsList += [string]$evalSeed
        }
        $argsList += @(
            "--episodes-per-seed", [string]$MODEL_SELECT_EPISODES,
            "--cotraining-pair", "${henRoundPath}::${eagleRoundPath}",
            "--episode-output", $episodeCsv,
            "--seed-output", $seedCsv,
            "--run-output", $runCsv,
            "--latex-output", $latexTex
        )
        Invoke-Python -ArgsList $argsList

        $runSummary = @(Import-Csv $runCsv)
        if ($runSummary.Count -ne 1) {
            throw "Expected exactly one run summary for seed $Seed round $round"
        }

        $captureRate = [double]$runSummary[0].capture_rate
        $avgTailDist = [double]$runSummary[0].avg_tail_dist
        $episodeLength = [double]$runSummary[0].episode_length

        $isBetter = $false
        if ($captureRate -lt $bestCaptureRate) {
            $isBetter = $true
        } elseif ($captureRate -eq $bestCaptureRate -and $avgTailDist -gt $bestAvgTailDist) {
            $isBetter = $true
        } elseif (
            $captureRate -eq $bestCaptureRate `
            -and $avgTailDist -eq $bestAvgTailDist `
            -and $episodeLength -gt $bestEpisodeLength
        ) {
            $isBetter = $true
        }

        if ($isBetter) {
            $bestPair = "${henRoundPath}::${eagleRoundPath}"
            $bestRound = $round
            $bestCaptureRate = $captureRate
            $bestAvgTailDist = $avgTailDist
            $bestEpisodeLength = $episodeLength
        }

        Remove-Item $episodeCsv, $seedCsv, $runCsv, $latexTex -ErrorAction SilentlyContinue
    }

    if ($null -eq $bestPair) {
        throw "Failed to select a best co-training checkpoint for seed $Seed"
    }

    Write-Host (
        "[Seed {0}] Best co-training round = {1} (capture_rate={2:F4}, avg_tail_dist={3:F3}, episode_length={4:F2})" `
        -f $Seed, $bestRound, $bestCaptureRate, $bestAvgTailDist, $bestEpisodeLength
    ) -ForegroundColor Green

    return $bestPair
}

Write-Host "=========================================================="
Write-Host "Starting Multi-Seed Training Pipeline (AeroPursuit)"
Write-Host "Hardware target: RTX 5080"
Write-Host "=========================================================="

$curriculumPairs = New-Object System.Collections.Generic.List[string]
$cotrainingPairs = New-Object System.Collections.Generic.List[string]

foreach ($seed in $SEEDS) {
    Write-Host "`n>>> Starting SEED $seed <<<" -ForegroundColor Cyan

    $currDir = "results/curriculum_seed_$seed"
    $coDir = "results/cotraining_seed_$seed"

    Write-Host "[Seed $seed] Training Curriculum Agent (Hen Stage 1)..." -ForegroundColor Yellow
    Invoke-Python -ArgsList @(
        "src/train_hen.py",
        "--seed", [string]$seed,
        "--total-steps", [string]$TOTAL_STEPS,
        "--save-dir", $currDir,
        "--device", $DEVICE
    )

    Write-Host "[Seed $seed] Training Curriculum Agent (Eagle Stage 2)..." -ForegroundColor Yellow
    $henModelPath = Join-Path $currDir "best_hen/best_model.zip"
    Invoke-Python -ArgsList @(
        "src/train_eagle.py",
        "--seed", [string]$seed,
        "--total-steps", [string]$TOTAL_STEPS,
        "--save-dir", $currDir,
        "--hen-model", $henModelPath,
        "--device", $DEVICE
    )

    $curriculumPairs.Add("${henModelPath}::$(Join-Path $currDir 'best_eagle/best_model.zip')")

    Write-Host "[Seed $seed] Training Co-training Baseline (from scratch)..." -ForegroundColor Yellow
    Invoke-Python -ArgsList @(
        "src/train_stage3.py",
        "--init-from-scratch",
        "--seed", [string]$seed,
        "--rounds", [string]$CO_ROUNDS,
        "--steps-per-round", [string]$CO_STEPS_PER_ROUND,
        "--save-dir", $coDir,
        "--device", $DEVICE
    )

    $bestCoPair = Select-BestCoTrainingPair -CoDir $coDir -Seed $seed -Rounds $CO_ROUNDS
    $cotrainingPairs.Add($bestCoPair)
}

Write-Host "`n=========================================================="
Write-Host "All training loops completed. Running final multi-seed evaluation..."
Write-Host "=========================================================="

$finalEvalArgs = @(
    "src/evaluate.py",
    "--method", "all"
)

foreach ($pair in $curriculumPairs) {
    $finalEvalArgs += @("--curriculum-pair", $pair)
}
foreach ($pair in $cotrainingPairs) {
    $finalEvalArgs += @("--cotraining-pair", $pair)
}

Invoke-Python -ArgsList $finalEvalArgs

Write-Host "=========================================================="
Write-Host "Experiments finished."
Write-Host "Check results/eval_run_summary.csv and results/latex_tables.tex."
Write-Host "=========================================================="
