#!/usr/bin/env python3
"""
Download a scene from HuggingFace, process with convert_to_unified.py, and upload to Cloudflare R2.

Usage:
    python process_and_upload.py --scene 0001
    python process_and_upload.py --scene 0001 --skip_download  # if already downloaded
    python process_and_upload.py --scene 0001 --skip_upload    # process only, no upload
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

# R2 Configuration
R2_ENDPOINT = "https://f29b8a79ec7fcd28ba93e118ad346921.r2.cloudflarestorage.com"
R2_ACCESS_KEY = "cffc851c37f414448fa429ca2b23f4c6"
R2_SECRET_KEY = "0db34fe73565d147f9c04021036203f7c11decd3e35deea4bc5d84dcc73aff3f"
R2_BUCKET = "szdec"

# HuggingFace Configuration
HF_REPO_ID = "kvuong2711/aerialmegadepth"


def get_r2_client():
    """Create and return R2 S3 client."""
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def download_scene(scene_id: str, output_dir: Path) -> Path:
    """Download a scene zip from HuggingFace and extract it.

    Args:
        scene_id: Scene identifier (e.g., "0001")
        output_dir: Directory to extract to

    Returns:
        Path to extracted scene directory
    """
    print(f"Downloading scene {scene_id} from HuggingFace...")

    zip_filename = f"{scene_id}.zip"

    # Download from HuggingFace
    zip_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=zip_filename,
        repo_type="dataset",
        local_dir=output_dir,
    )
    zip_path = Path(zip_path)
    print(f"Downloaded to: {zip_path}")

    # Extract zip
    extract_dir = output_dir / "data"
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting to: {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Remove zip file
    zip_path.unlink()
    print(f"Removed zip file")

    scene_dir = extract_dir / scene_id
    if not scene_dir.exists():
        # Maybe it extracted directly
        scene_dir = extract_dir

    return scene_dir


def process_scene(scene_dir: Path, output_dir: Path) -> Path:
    """Process scene with convert_to_unified.py.

    Args:
        scene_dir: Path to scene directory (e.g., data/0001)
        output_dir: Path to output unified format directory

    Returns:
        Path to unified output directory
    """
    # Find the localized_dense_metric directory
    input_dir = scene_dir / "sfm_output_localization" / "sfm_superpoint+superglue" / "localized_dense_metric"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print(f"Processing scene from: {input_dir}")
    print(f"Output to: {output_dir}")

    # Get the directory containing this script
    script_dir = Path(__file__).parent
    convert_script = script_dir / "convert_to_unified.py"

    # Run convert_to_unified.py
    cmd = [
        sys.executable,
        str(convert_script),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--copy_images",  # Copy instead of symlink for upload
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    return output_dir


def upload_to_r2(local_dir: Path, scene_id: str, prefix: str = "aerial_megadepth"):
    """Upload unified scene to R2.

    Args:
        local_dir: Local directory containing unified format
        scene_id: Scene identifier for R2 path
        prefix: R2 prefix (folder)
    """
    s3 = get_r2_client()

    r2_prefix = f"{prefix}/{scene_id}"
    print(f"Uploading to R2: s3://{R2_BUCKET}/{r2_prefix}/")

    files_uploaded = 0
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            relative_path = local_path.relative_to(local_dir)
            r2_key = f"{r2_prefix}/{relative_path}"

            print(f"  Uploading: {relative_path}")
            s3.upload_file(str(local_path), R2_BUCKET, r2_key)
            files_uploaded += 1

    print(f"Uploaded {files_uploaded} files to s3://{R2_BUCKET}/{r2_prefix}/")


def main():
    parser = argparse.ArgumentParser(
        description="Download, process, and upload AerialMegaDepth scene to R2"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene ID to process (e.g., 0001)"
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("./work"),
        help="Working directory for downloads and processing"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for unified format (default: work_dir/unified/SCENE)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download, use existing data in work_dir"
    )
    parser.add_argument(
        "--skip_process",
        action="store_true",
        help="Skip processing, use existing unified output"
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Skip upload to R2"
    )
    parser.add_argument(
        "--r2_prefix",
        type=str,
        default="aerial_megadepth",
        help="R2 prefix/folder (default: aerial_megadepth)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete downloaded and processed data after successful upload"
    )

    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    scene_id = args.scene
    output_dir = args.output_dir or (work_dir / "unified" / scene_id)

    print(f"=" * 60)
    print(f"Processing scene: {scene_id}")
    print(f"Work directory: {work_dir}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)

    # Step 1: Download
    if not args.skip_download:
        scene_dir = download_scene(scene_id, work_dir)
    else:
        scene_dir = work_dir / "data" / scene_id
        if not scene_dir.exists():
            print(f"Error: Scene directory not found: {scene_dir}")
            sys.exit(1)
        print(f"Using existing scene data: {scene_dir}")

    # Step 2: Process
    if not args.skip_process:
        process_scene(scene_dir, output_dir)
    else:
        if not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}")
            sys.exit(1)
        print(f"Using existing unified output: {output_dir}")

    # Step 3: Upload
    if not args.skip_upload:
        upload_to_r2(output_dir, scene_id, args.r2_prefix)
    else:
        print("Skipping upload to R2")

    # Step 4: Cleanup
    if args.cleanup and not args.skip_upload:
        import shutil
        print("\nCleaning up...")

        # Delete downloaded scene data immediately
        downloaded_scene_dir = work_dir / "data" / scene_id
        if downloaded_scene_dir.exists():
            print(f"  Removing downloaded data: {downloaded_scene_dir}")
            shutil.rmtree(downloaded_scene_dir)

        # Also clean up the data parent dir if empty
        data_dir = work_dir / "data"
        if data_dir.exists() and not any(data_dir.iterdir()):
            data_dir.rmdir()

        # Delete unified output
        if output_dir.exists():
            print(f"  Removing processed data: {output_dir}")
            shutil.rmtree(output_dir)

        # Also clean up the unified parent dir if empty
        unified_dir = work_dir / "unified"
        if unified_dir.exists() and not any(unified_dir.iterdir()):
            unified_dir.rmdir()

        # Delete the entire work_dir if it's empty (for per-scene work dirs)
        if work_dir.exists() and not any(work_dir.iterdir()):
            print(f"  Removing empty work directory: {work_dir}")
            work_dir.rmdir()

        print("Cleanup complete!")

    print(f"\nDone! Scene {scene_id} processed and uploaded.")
    print(f"R2 location: s3://{R2_BUCKET}/{args.r2_prefix}/{scene_id}/")


if __name__ == "__main__":
    main()
