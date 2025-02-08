# Step 1: Run the pip show command and capture the output
$pipShowOutput = & pip show pybind11

# Extract the Location line using Select-String
$locationLine = $pipShowOutput | Select-String -Pattern '^Location:'

# Extract the path from the Location line
$locationPath = $locationLine -replace '^Location:\s*', ''

# Step 2: Run CMake commands in the 'build' directory
if (-Not (Test-Path -Path "./build")) {
    New-Item -ItemType Directory -Path "./build"
}
Set-Location -Path "./build"

cmake .. -DCMAKE_PREFIX_PATH="$locationPath"
cmake --build . --config Release

# Step 3: Copy the .pyd file to the Modelling 2B folder using relative paths
$sourcePath = ".\Release\simulation.cp311-win_amd64.pyd"
$destinationPath = "..\..\..\..\simulation.cp311-win_amd64.pyd"

Copy-Item -Path $sourcePath -Destination $destinationPath

# Return to the simulation folder
Set-Location -Path ..

# Resolve absolute paths for better reliability
$existingFilePath = Resolve-Path "..\..\..\simulation.pyd" -ErrorAction SilentlyContinue
$newFilePath = Resolve-Path "..\..\..\simulation.cp311-win_amd64.pyd" -ErrorAction SilentlyContinue
$finalFilePath = "..\..\..\simulation.pyd"

Write-Output "Existing File Path: $existingFilePath"
Write-Output "New File Path: $newFilePath"
Write-Output "Final File Path: $finalFilePath"

# Step 4: Determine the highest number from existing backups and generate the new backup file name
$archiveDir = "..\..\..\Archive\Archive Simulation\Old simulations"
$existingFiles = Get-ChildItem -Path $archiveDir -Filter "simulation*.pyd"
$highestNumber = $existingFiles | ForEach-Object {
    if ($_ -match "simulation(?:_old)?(\d+).pyd") { [int]$matches[1] }
} | Sort-Object -Descending | Select-Object -First 1
$nextNumber = $highestNumber + 1
$backupFilePath = "$archiveDir\simulation$nextNumber.pyd"

Write-Output "Backup File Path: $backupFilePath"

# Backup the existing simulation.pyd by copying instead of renaming
if ($existingFilePath) {
    Copy-Item -Path $existingFilePath -Destination $backupFilePath -Force
    Write-Output "Backup created: $backupFilePath"
} else {
    Write-Output "Existing simulation.pyd file not found, skipping backup."
}

# Delete existing simulation.pyd if it exists
if (Test-Path $finalFilePath) {
    Remove-Item -Path $finalFilePath -Force
    Write-Output "Existing simulation.pyd file deleted."
}

# Copy the new file to the final file path
if ($newFilePath) {
    try {
        Copy-Item -Path $newFilePath -Destination $finalFilePath -Force
        Write-Output "New simulation.pyd file copied."
    } catch {
        Write-Output "Error copying file: $_"
    }
} else {
    Write-Output "New simulation.cp311-win_amd64.pyd file not found."
}

# Remove the simulation.cp311-win_amd64.pyd file after copying it
if (Test-Path $newFilePath) {
    Remove-Item -Path $newFilePath -Force
    Write-Output "Temporary simulation.cp311-win_amd64.pyd file deleted."
}
