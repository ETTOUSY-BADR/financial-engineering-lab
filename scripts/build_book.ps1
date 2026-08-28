[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$courseDir = Join-Path $repoRoot "course"

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    throw "pdflatex is required. Install MiKTeX or TeX Live and place it on PATH."
}

Push-Location $courseDir
try {
    pdflatex -interaction=nonstopmode -halt-on-error quant_finance_book.tex
    pdflatex -interaction=nonstopmode -halt-on-error quant_finance_book.tex
}
finally {
    Pop-Location
}

$pdf = Join-Path $courseDir "quant_finance_book.pdf"
Write-Output "Built $pdf"
