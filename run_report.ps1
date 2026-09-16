<#
.SYNOPSIS
    Generate an equity research report for one ticker - load .env, then run
    the pipeline. Interactive (prompts for a ticker if you don't pass one) or
    scriptable (pass -Ticker to run non-interactively, e.g. from another script).

.DESCRIPTION
    This is a convenience wrapper around `python -m eq_report`. It exists
    because every real (OpenRouter-backed) run needs the same few steps every
    time: load .env into the process environment (each PowerShell invocation
    starts with a clean environment, so .env has to be re-read every run),
    optionally skip provider endpoints that are unreachable from wherever
    you're running this, and optionally turn on the paid opt-in features
    (gap research, the data-freshness check, higher planning reasoning
    effort). Everything below is off unless you ask for it, so a plain
    `.\run_report.ps1 -Ticker NVDA` costs exactly what the base pipeline
    already costs - nothing extra is silently enabled.

.PARAMETER Ticker
    The ticker to research, e.g. NVDA. If omitted, you'll be prompted.

.PARAMETER ReportDate
    As-of date, YYYY-MM-DD. Defaults to today.

.PARAMETER OutputDir
    Where run artefacts go. Defaults to .\output (same as the pipeline's own
    default), so successive runs accumulate under output\runs\<run_id>\.

.PARAMETER SkipRealProviders
    Unset EQR_MEGADATA_BASE_URL / EQR_MEGADATA_API_KEY
    before running, so the pipeline falls straight back to its mock
    providers instead of trying (and, if the host is unreachable, hanging
    for several minutes on) a real data provider. Use this when running
    somewhere those endpoints are not reachable - e.g. this sandbox, where
    they are private-LAN addresses. Does not touch EQR_MODEL_API_KEY, so the
    LLM-backed stages (planning, agents, synthesis, QA) still run for real.

.PARAMETER EnableGapResearch
    Turn on live web-search gap-filling (EQR_RESEARCH_DATA_GAPS=true) for
    this run. Adds one model call per disclosed gap (capped by
    -MaxGapsResearched) - a genuine extra cost, off by default.

.PARAMETER MaxGapsResearched
    Cap on how many gaps -EnableGapResearch will pay to research. Default 8
    (the pipeline's own default); only meaningful together with
    -EnableGapResearch.

.PARAMETER CheckFreshness
    Turn on the live data-freshness check (EQR_CHECK_DATA_FRESHNESS=true)
    for this run - one extra model call, verifying the dataset is anchored
    on the latest publicly reported period. Off by default.

.PARAMETER VerifyMetricConflicts
    Turn on live web verification (EQR_VERIFY_METRIC_CONFLICTS=true) of a
    metric two sources disagree on: the model must find the real value from
    an actual, dated source before QA will unblock on it - it can never just
    pick whichever candidate looks more plausible. One extra model call per
    genuine conflict (capped), off by default.

.PARAMETER PlanningEffort
    Reasoning effort for the planning call only ("high", "medium", or
    "low"). Every other stage is unaffected. Omit to use the model's default
    effort.

.EXAMPLE
    .\run_report.ps1
    Prompts for a ticker, uses today's date, runs with whatever .env already
    has configured.

.EXAMPLE
    .\run_report.ps1 -Ticker NVDA -SkipRealProviders
    Runs NVDA right now, skipping any real market-data/fundamentals provider
    so it can't hang on an unreachable host - what this sandbox needs.

.EXAMPLE
    .\run_report.ps1 -Ticker AMD -ReportDate 2026-09-02 -EnableGapResearch -CheckFreshness
    A fuller run with both opt-in live-research features turned on.
#>
[CmdletBinding()]
param(
    [string]$Ticker,
    [string]$ReportDate = (Get-Date -Format "yyyy-MM-dd"),
    [string]$OutputDir = "output",
    [switch]$SkipRealProviders,
    [switch]$EnableGapResearch,
    [int]$MaxGapsResearched,
    [switch]$CheckFreshness,
    [switch]$VerifyMetricConflicts,
    [switch]$OnlineSources,
    [switch]$TechnicalAppendix,
    [switch]$CompactReport,
    [ValidateSet("high", "medium", "low")]
    [string]$PlanningEffort
)

$ErrorActionPreference = "Stop"

if (-not $Ticker) {
    $Ticker = Read-Host "Ticker to research (e.g. NVDA)"
}
$Ticker = $Ticker.Trim().ToUpper()
if (-not $Ticker) {
    Write-Error "A ticker is required."
    exit 1
}

$repoRoot = $PSScriptRoot
$envFile = Join-Path $repoRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -match '^\s*[A-Za-z_]+=' -and $_ -notmatch '^\s*#' } |
        ForEach-Object {
            $key, $value = $_.Split('=', 2)
            Set-Item -Path "Env:$key" -Value $value
        }
} else {
    Write-Warning "No .env found at $envFile - running with only this shell's existing environment."
}

if ($SkipRealProviders) {
    Remove-Item Env:\EQR_MEGADATA_BASE_URL -ErrorAction SilentlyContinue
    Remove-Item Env:\EQR_MEGADATA_API_KEY -ErrorAction SilentlyContinue
}
if ($EnableGapResearch) {
    $env:EQR_RESEARCH_DATA_GAPS = "true"
    if ($MaxGapsResearched) {
        $env:EQR_RESEARCH_DATA_GAPS_MAX = "$MaxGapsResearched"
    }
}
if ($CheckFreshness) {
    $env:EQR_CHECK_DATA_FRESHNESS = "true"
}
if ($VerifyMetricConflicts) {
    $env:EQR_VERIFY_METRIC_CONFLICTS = "true"
}
if ($OnlineSources) {
    $env:EQR_ONLINE_SOURCES = "true"
}
if ($TechnicalAppendix) {
    $env:EQR_TECHNICAL_APPENDIX = "true"
}
if ($CompactReport) {
    $env:EQR_COMPACT_REPORT = "true"
}
if ($PlanningEffort) {
    $env:EQR_PLANNING_REASONING_EFFORT = $PlanningEffort
}

$modelConfigured = [bool]($env:EQR_MODEL_API_KEY -or $env:OPENROUTER_API_KEY)
Write-Host "Ticker            : $Ticker"
Write-Host "Report date       : $ReportDate"
Write-Host "Output dir        : $OutputDir"
Write-Host "Model configured  : $modelConfigured$(if (-not $modelConfigured) { ' (mock/deterministic run - no API key set)' })"
Write-Host "Skip real providers: $([bool]$SkipRealProviders)"
Write-Host "Gap research      : $([bool]$EnableGapResearch)"
Write-Host "Freshness check   : $([bool]$CheckFreshness)"
Write-Host "Verify conflicts  : $([bool]$VerifyMetricConflicts)"
Write-Host "Online sources    : $([bool]$OnlineSources)"
Write-Host "Technical appendix: True (always generated)"
Write-Host "Compact report     : True (always generated)"
Write-Host "Planning effort   : $(if ($PlanningEffort) { $PlanningEffort } else { '(default)' })"
Write-Host ""

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "python was not found on PATH. Activate the project's virtualenv (or install the " +
        "requirements with 'pip install -r requirements.txt') before running this script."
    exit 1
}

& python -m eq_report --ticker $Ticker --report-date $ReportDate --output-dir $OutputDir
exit $LASTEXITCODE
