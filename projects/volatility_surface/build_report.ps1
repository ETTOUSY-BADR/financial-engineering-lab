[CmdletBinding()]
param(
    [string]$PdfLaTeX = "pdflatex"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = $PSScriptRoot
Push-Location $projectRoot
try {
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error report.tex
    if ($LASTEXITCODE -ne 0) {
        throw "First report compilation failed with exit code $LASTEXITCODE."
    }
    & $PdfLaTeX -interaction=nonstopmode -halt-on-error report.tex
    if ($LASTEXITCODE -ne 0) {
        throw "Second report compilation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
