param(
    [string]$DatasetPath = "data",
    [string]$Colmap = "colmap"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DatasetPath)) {
    throw "Dataset path not found: $DatasetPath"
}

$imagePath = Join-Path $DatasetPath "images"
$colmapPath = Join-Path $DatasetPath "colmap_sparse_only"
$sparsePath = Join-Path $colmapPath "sparse"

New-Item -ItemType Directory -Force -Path $sparsePath | Out-Null

Write-Host "=== Step 1: Feature Extraction ==="
& $Colmap feature_extractor `
    --database_path (Join-Path $colmapPath "database.db") `
    --image_path $imagePath `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1

Write-Host "=== Step 2: Feature Matching ==="
& $Colmap exhaustive_matcher `
    --database_path (Join-Path $colmapPath "database.db")

Write-Host "=== Step 3: Sparse Reconstruction ==="
& $Colmap mapper `
    --database_path (Join-Path $colmapPath "database.db") `
    --image_path $imagePath `
    --output_path $sparsePath

Write-Host "=== Done ==="
Write-Host "Sparse model: $(Join-Path $sparsePath '0')"
