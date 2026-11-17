param([Parameter(Mandatory=$true)][string]$RepoUrl)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($RepoUrl -notmatch '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$') { throw 'Use the confirmed GitHub HTTPS repository URL without credentials.' }
Set-Location -LiteralPath $projectRoot
$stateDir=Join-Path $projectRoot '.sync-state'
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
$logPath=Join-Path $stateDir 'sync.log'
Start-Transcript -Path $logPath -Append | Out-Null
try {
$env:GIT_TERMINAL_PROMPT='0'
$env:GCM_INTERACTIVE='Never'
$actualRoot=(& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath($actualRoot) -ne $projectRoot) { throw 'Initialize and connect a dedicated MCCAI repository first. Parent repositories will not be used.' }
$remote=(& git remote get-url origin)
if ($LASTEXITCODE -ne 0 -or $remote.TrimEnd('/').Replace('.git','') -ne $RepoUrl.TrimEnd('/').Replace('.git','')) { throw 'Origin differs from the confirmed GitHub repository.' }
$branch=(& git symbolic-ref --short HEAD)
if ($LASTEXITCODE -ne 0) { throw 'Detached HEAD is not supported.' }
& git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Pre-existing staged changes found; automatic sync stopped for review.' }
# Ignore rules exclude original data, per-sample outputs, model files and credentials.
& git add -- .gitignore README.md research scripts docs
if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
$files=@(& git -c core.quotepath=false diff --cached --name-only)
foreach ($name in $files) {
    if ($name -match '(^|/)(data|datasets|outputs|manifests|splits|cache|weights|checkpoints|\.openai|\.aws|\.codex|\.sync-state)/|(^|/)\.env($|\.)|\.(pt|pth|safetensors|npy|npz|parquet|zip)$') { throw "Excluded artifact staged: $name. Review the index manually." }
    $fullPath=Join-Path $projectRoot $name
    if ((Test-Path -LiteralPath $fullPath) -and (Get-Item -LiteralPath $fullPath).Length -gt 10MB) { throw "Unexpected large staged file: $name. Review manually." }
}
& git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    & git commit -m ('chore: WIP research auto backup '+(Get-Date -Format 'yyyy-MM-dd HH:mm')) -m 'Automatic backup only; this task does not run tests or certify completion.'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed; check Git identity and staged changes.' }
} elseif ($LASTEXITCODE -ne 0) { throw 'Index inspection failed.' }
& git push origin $branch
if ($LASTEXITCODE -ne 0) { throw 'Push failed; verify authentication or remote changes. No force-push or automatic merge is performed.' }
Write-Output ('GitHub synchronized at '+(Get-Date -Format o))
@{last_success=(Get-Date -Format o);branch=$branch;repository=$RepoUrl;commit=(& git rev-parse HEAD)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateDir 'last-success.json') -Encoding UTF8
} finally { Stop-Transcript | Out-Null }
