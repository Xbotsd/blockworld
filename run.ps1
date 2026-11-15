# Verify: tests -> study example (MC + Morris) -> demo report
# Usage: powershell -ExecutionPolicy Bypass -File .\run.ps1
$ErrorActionPreference = "Stop"

$py = "C:\Users\lei'shen\.dsh\dsh-runtimes\dsh-next-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $py)) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { throw "Python interpreter not found" }
}

Set-Location $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

Write-Host "== tests ==" -ForegroundColor Cyan
& $py -m unittest discover -s tests

Write-Host "== data template ==" -ForegroundColor Cyan
& $py -m blockworld template data\site

Write-Host "== study example (Monte Carlo + Morris) ==" -ForegroundColor Cyan
& $py -m blockworld study examples\study_landing.json --out out_study --mc 60 --morris 6

Write-Host "== demo ==" -ForegroundColor Cyan
& $py -m blockworld demo --out out --years 55 --animals 6

Write-Host ""
Write-Host "report: $PSScriptRoot\out\report.html"
Write-Host "study : $PSScriptRoot\out_study\study.json"
