[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$projects = @(
    "projects/regime_risk/run_research.py",
    "projects/yield_curve_dynamics/run_research.py",
    "projects/factor_allocation/run_research.py",
    "projects/volatility_surface/run_research.py",
    "projects/volatility_surface/run_multidate.py"
)

Push-Location $repoRoot
try {
    & $Python -m unittest discover -s tests -v
    foreach ($project in $projects) {
        Write-Output "Running $project"
        & $Python $project
    }
}
finally {
    Pop-Location
}
