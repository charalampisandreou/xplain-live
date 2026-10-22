<#
.SYNOPSIS
  Build the Studio image, push it to ACR and deploy infra/main.json (environment + app).

.DESCRIPTION
  ACR Tasks (az acr build) is paused on credit subscriptions (TasksOperationsNotAllowed), so the image is
  built with local Docker and pushed. The tag is the git short sha, so the working tree must be clean.
  The model and data/bundle_seed3001.json are git-ignored and copied from this checkout.
  infra/main.json creates or updates the environment cae-xplain-std and the app xplain-live; the app
  pulls with the user-assigned identity id-xplain-live (AcrPull), so there are no registry passwords or
  tokens. The what-if preview is printed as resource IDs only: its full output contains the workspace key.
  Needs: az login, Docker running.

.EXAMPLE
  ./scripts/deploy_azure.ps1
#>
[CmdletBinding()]
param(
    [string]$ResourceGroup = 'rg-plhack-de',
    [ValidateSet('germanywestcentral')]
    [string]$Location = 'germanywestcentral',
    [string]$Registry = 'xplainlive',
    [string]$AppName = 'xplain-live',
    [string]$Deployment = 'xplain-live'
)
$ErrorActionPreference = 'Stop'

function Invoke-Native {
    # run a native command (az, docker, git) and stop on a non-zero exit code
    $exe, $rest = $args
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "$exe $($rest -join ' ') failed (exit $LASTEXITCODE)" }
}

$root = Split-Path $PSScriptRoot -Parent
$template = Join-Path $root 'infra/main.json'

Write-Host "== preflight"
Invoke-Native az account show --query "{subscription:name, tenant:tenantId}" -o table
if (Invoke-Native git -C $root status --porcelain) { throw 'working tree is not clean: commit first (the tag is the git short sha)' }
$tag = Invoke-Native git -C $root rev-parse --short HEAD
$image = "$Registry.azurecr.io/xplain-live:$tag"
foreach ($f in 'models/xpass_v2/xpass_v2.joblib', 'models/xpass_v2/feature_list.json', 'data/bundle_seed3001.json') {
    if (-not (Test-Path (Join-Path $root $f))) { throw "missing $f (the image copies it from this checkout)" }
}

Write-Host "== build and push $image"
# a single linux/amd64 manifest, no attestation index
Invoke-Native docker build --platform linux/amd64 --provenance=false --sbom=false -t $image $root
Invoke-Native az acr login -n $Registry --only-show-errors   # token goes to Docker's credential store
Invoke-Native docker push $image

Write-Host "== what-if (resource IDs only)"
Invoke-Native az deployment group what-if -g $ResourceGroup -n $Deployment --template-file $template `
    --parameters location=$Location image=$image appName=$AppName --result-format ResourceIdOnly
if ((Read-Host "Deploy $image? (y/n)") -ne 'y') { Write-Host 'not deployed'; return }

Write-Host "== deploy"
$fqdn = Invoke-Native az deployment group create -g $ResourceGroup -n $Deployment --template-file $template `
    --parameters location=$Location image=$image appName=$AppName --query properties.outputs.fqdn.value -o tsv
Write-Host "== https://$fqdn/"
(Invoke-WebRequest "https://$fqdn/health" -UseBasicParsing -TimeoutSec 240).Content
