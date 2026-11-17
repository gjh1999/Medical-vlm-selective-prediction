param([Parameter(Mandatory=$true)][string]$RepoUrl,[ValidateRange(5,1440)][int]$Minutes=30)
$ErrorActionPreference='Stop'
if ($RepoUrl -notmatch '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$') { throw 'Provide a GitHub HTTPS repository URL without credentials.' }
$sync=Join-Path $PSScriptRoot 'sync-github.ps1'
# Require a successful real push before enabling unattended execution.
& $sync -RepoUrl $RepoUrl
if (-not $?) { throw 'First sync failed; no schedule registered.' }
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -File "'+$sync+'" -RepoUrl "'+$RepoUrl+'"')
$trigger=New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes($Minutes) -RepetitionInterval (New-TimeSpan -Minutes $Minutes)
$settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'MCCAI-Research-GitHub-Sync' -Action $action -Trigger $trigger -Settings $settings -Description 'Sync research code and summaries; exclude original data and model artifacts.'
