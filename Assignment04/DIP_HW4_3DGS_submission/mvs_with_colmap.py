import os
import subprocess
import argparse

# Allow COLMAP (Qt-based) to run on headless servers without an X display.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def run_colmap(args):
    colmap_bin = os.environ.get('COLMAP_BIN', 'colmap')
    subprocess.run([colmap_bin, *args], check=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    args = parser.parse_args()
    data_dir = args.data_dir
    use_gpu = os.environ.get('COLMAP_USE_GPU', '0')
    gpu_index = os.environ.get('COLMAP_GPU_INDEX', '0')
    num_threads = os.environ.get('COLMAP_NUM_THREADS', '8')

    # Feature extraction with shared intrinsics (assume it's the same camera)
    run_colmap(['feature_extractor', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'PINHOLE', '--FeatureExtraction.use_gpu', use_gpu, '--FeatureExtraction.gpu_index', gpu_index, '--FeatureExtraction.num_threads', num_threads])

    # Feature matching
    run_colmap(['exhaustive_matcher', '--database_path', os.path.join(data_dir, 'database.db'), '--FeatureMatching.use_gpu', use_gpu, '--FeatureMatching.gpu_index', gpu_index, '--FeatureMatching.num_threads', num_threads, '--ExhaustiveMatching.block_size', '20'])

    # Create sparse reconstruction folder
    os.makedirs(os.path.join(data_dir, 'sparse'), exist_ok=True)

    # Sparse reconstruction
    run_colmap(['mapper', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--output_path', os.path.join(data_dir, 'sparse')])

    # Convert binary model to text format
    os.makedirs(os.path.join(data_dir, 'sparse', '0_text'), exist_ok=True)
    run_colmap(['model_converter', '--input_path', os.path.join(data_dir, 'sparse', '0'), '--output_path', os.path.join(data_dir, 'sparse', '0_text'), '--output_type', 'TXT'])

    print("COLMAP multi-view stereo pipeline completed successfully!")
    print("Sparse 3D reconstruction saved in:", os.path.join(data_dir, 'sparse', '0_text'))
    
