param(
    [string]$DatasetPath = "data",
    [string]$Colmap = "colmap"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DatasetPath)) {
    throw "Dataset path not found: $DatasetPath"
}

$imagePath = Join-Path $DatasetPath "images"
$colmapPath = Join-Path $DatasetPath "colmap"
$sparsePath = Join-Path $colmapPath "sparse"
$densePath = Join-Path $colmapPath "dense"

New-Item -ItemType Directory -Force -Path $sparsePath | Out-Null
New-Item -ItemType Directory -Force -Path $densePath | Out-Null

Write-Host "=== Step 1: Feature Extraction ==="
& $Colmap feature_extractor `
    --database_path (Join-Path $colmapPath "database.db") `
    --image_path $imagePath `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1

Write-Host "=== Step 2: Feature Matching ==="
& $Colmap exhaustive_matcher `
    --database_path (Join-Path $colmapPath "database.db")

Write-Host "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
& $Colmap mapper `
    --database_path (Join-Path $colmapPath "database.db") `
    --image_path $imagePath `
    --output_path $sparsePath

Write-Host "=== Step 4: Image Undistortion ==="
& $Colmap image_undistorter `
    --image_path $imagePath `
    --input_path (Join-Path $sparsePath "0") `
    --output_path $densePath

Write-Host "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
& $Colmap patch_match_stereo `
    --workspace_path $densePath

Write-Host "=== Step 6: Stereo Fusion ==="
& $Colmap stereo_fusion `
    --workspace_path $densePath `
    --output_path (Join-Path $densePath "fused.ply")

Write-Host "=== Done ==="
Write-Host "Sparse: $(Join-Path $sparsePath '0')"
Write-Host "Dense:  $(Join-Path $densePath 'fused.ply')"
