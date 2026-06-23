# init_volume.ps1
# Using single quotes to prevent PowerShell from misinterpreting '!' and '@'

$HOST_DATA_PATH = 'D:\!staging\@Mestrado\PCBA-AOI\output_sliced'
$VOLUME_NAME = "rfdetr_dataset"

# Robust check: Ask Docker to list volumes matching the exact name
$existingVolume = docker volume ls -q --filter "name=^${VOLUME_NAME}$"

if ($existingVolume) {
    Write-Host "Volume '$VOLUME_NAME' already exists." -ForegroundColor Yellow
    $replace = Read-Host "Do you want to DELETE the existing volume and REPLACE it with the new dataset? (y/n)"
    
    if ($replace -eq 'y') {
        Write-Host "Deleting existing volume..." -ForegroundColor Cyan
        docker volume rm $VOLUME_NAME
    } else {
        Write-Host "Aborting. Volume left untouched." -ForegroundColor Yellow
        exit
    }
}

Write-Host "Creating Docker volume: $VOLUME_NAME" -ForegroundColor Green
docker volume create $VOLUME_NAME

Write-Host "Copying data from host to Docker volume..." -ForegroundColor Cyan
Write-Host "Source: $HOST_DATA_PATH" -ForegroundColor Gray
Write-Host "This might take a few minutes depending on dataset size." -ForegroundColor Yellow

# Run a temporary Alpine container to wipe the destination and copy fresh data
docker run --rm `
    -v "${HOST_DATA_PATH}:/src" `
    -v "${VOLUME_NAME}:/dest" `
    alpine `
    sh -c "rm -rf /dest/* && cp -a /src/. /dest/"

Write-Host "Data transfer complete!" -ForegroundColor Green