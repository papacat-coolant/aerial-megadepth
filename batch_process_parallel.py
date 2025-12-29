#!/usr/bin/env python3
"""
Parallel batch processing for AerialMegaDepth scenes from HuggingFace.

Downloads scenes from HuggingFace, processes them, uploads to R2, and cleans up.

Usage:
    # Download and process scenes 0000-0065 (00*.zip pattern)
    python batch_process_parallel.py --allow_patterns "00*.zip"

    # Download and process scenes 0100-0199 (01*.zip pattern)
    python batch_process_parallel.py --allow_patterns "01*.zip"

    # Download specific files
    python batch_process_parallel.py --allow_patterns "0001.zip" "0002.zip" "0003.zip"

    # Download all zip files
    python batch_process_parallel.py --allow_patterns "*.zip"

    # Use more workers for faster processing
    python batch_process_parallel.py --allow_patterns "00*.zip" --workers 8

    # Skip upload (only download and process)
    python batch_process_parallel.py --allow_patterns "00*.zip" --skip_upload

    # Don't cleanup after processing (keep downloaded files)
    python batch_process_parallel.py --allow_patterns "00*.zip" --no_cleanup
"""

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)


# HuggingFace Configuration
HF_REPO_ID = "kvuong2711/aerialmegadepth"


@dataclass
class ProcessResult:
    scene_id: str
    success: bool
    duration: float
    error: Optional[str] = None


# Lock for thread-safe printing
print_lock = Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


def get_scene_files_from_hf(allow_patterns: list[str]) -> list[str]:
    """Get list of scene zip files from HuggingFace matching patterns.

    Args:
        allow_patterns: List of glob patterns (e.g., ["00*.zip", "01*.zip"])

    Returns:
        List of scene IDs (e.g., ["0001", "0002", ...])
    """
    print(f"Fetching file list from HuggingFace: {HF_REPO_ID}")

    # Get all files in the repo
    all_files = list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")

    # Filter to only zip files matching patterns
    matched_files = []
    for filename in all_files:
        # Only consider .zip files that look like scene IDs (4 digits)
        if not filename.endswith('.zip'):
            continue
        if not re.match(r'^\d{4}\.zip$', filename):
            continue

        # Check if file matches any pattern
        for pattern in allow_patterns:
            if fnmatch.fnmatch(filename, pattern):
                matched_files.append(filename)
                break

    # Extract scene IDs from filenames
    scene_ids = sorted([f.replace('.zip', '') for f in matched_files])

    print(f"Found {len(scene_ids)} scenes matching patterns: {allow_patterns}")
    if scene_ids:
        print(f"  Scenes: {scene_ids[0]} - {scene_ids[-1]}")

    return scene_ids


def download_scene(scene_id: str, work_dir: Path) -> Path:
    """Download a scene zip from HuggingFace and extract it.

    Args:
        scene_id: Scene identifier (e.g., "0001")
        work_dir: Working directory

    Returns:
        Path to extracted scene directory
    """
    zip_filename = f"{scene_id}.zip"

    # Download from HuggingFace
    zip_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=zip_filename,
        repo_type="dataset",
        local_dir=work_dir,
    )
    zip_path = Path(zip_path)

    # Extract zip
    extract_dir = work_dir / "data"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Remove zip file
    zip_path.unlink()

    scene_dir = extract_dir / scene_id
    return scene_dir


def process_scene(
    scene_id: str,
    work_dir: Path,
    script_dir: Path,
    cleanup: bool = True,
    skip_upload: bool = False,
) -> ProcessResult:
    """Process a single scene: download, convert, upload, cleanup.

    Args:
        scene_id: Scene identifier (e.g., "0001")
        work_dir: Working directory
        script_dir: Directory containing convert_to_unified.py
        cleanup: Whether to cleanup after processing
        skip_upload: Skip upload step

    Returns:
        ProcessResult with success status and timing
    """
    start_time = time.time()
    scene_work_dir = work_dir / scene_id

    safe_print(f"[{scene_id}] Starting processing...")

    try:
        # Step 1: Download from HuggingFace
        safe_print(f"[{scene_id}] Downloading from HuggingFace...")
        scene_dir = download_scene(scene_id, scene_work_dir)
        safe_print(f"[{scene_id}] Downloaded and extracted to: {scene_dir}")

        # Step 2: Process with convert_to_unified.py
        input_dir = scene_dir / "sfm_output_localization" / "sfm_superpoint+superglue" / "localized_dense_metric"

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir = scene_work_dir / "unified" / scene_id
        output_dir.mkdir(parents=True, exist_ok=True)

        convert_script = script_dir / "convert_to_unified.py"

        safe_print(f"[{scene_id}] Converting to unified format...")
        cmd = [
            sys.executable,
            str(convert_script),
            "--input_dir", str(input_dir),
            "--output_dir", str(output_dir),
            "--copy_images",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Step 3: Upload to R2 (if not skipped)
        if not skip_upload:
            safe_print(f"[{scene_id}] Uploading to R2...")
            upload_script = script_dir / "process_and_upload.py"

            # Use process_and_upload.py with skip_download and skip_process
            cmd = [
                sys.executable,
                str(upload_script),
                "--scene", scene_id,
                "--work_dir", str(scene_work_dir),
                "--skip_download",
                "--skip_process",
            ]
            if cleanup:
                cmd.append("--cleanup")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        else:
            safe_print(f"[{scene_id}] Skipping upload to R2")

        # Step 4: Cleanup (if requested and not handled by process_and_upload.py)
        if cleanup and skip_upload:
            safe_print(f"[{scene_id}] Cleaning up...")
            if scene_work_dir.exists():
                shutil.rmtree(scene_work_dir)

        duration = time.time() - start_time
        safe_print(f"[{scene_id}] Completed successfully in {duration:.1f}s")
        return ProcessResult(scene_id=scene_id, success=True, duration=duration)

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_msg = e.stderr[:500] if e.stderr else str(e)
        safe_print(f"[{scene_id}] Failed after {duration:.1f}s: {error_msg}")

        # Cleanup on failure if requested
        if cleanup and scene_work_dir.exists():
            shutil.rmtree(scene_work_dir)

        return ProcessResult(scene_id=scene_id, success=False, duration=duration, error=error_msg)

    except Exception as e:
        duration = time.time() - start_time
        safe_print(f"[{scene_id}] Error after {duration:.1f}s: {e}")

        # Cleanup on failure if requested
        if cleanup and scene_work_dir.exists():
            shutil.rmtree(scene_work_dir)

        return ProcessResult(scene_id=scene_id, success=False, duration=duration, error=str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch processing for AerialMegaDepth scenes from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process scenes matching 00*.zip (0000-0065)
  python batch_process_parallel.py --allow_patterns "00*.zip"

  # Download multiple patterns
  python batch_process_parallel.py --allow_patterns "00*.zip" "01*.zip"

  # Download specific scenes
  python batch_process_parallel.py --allow_patterns "0001.zip" "0002.zip"

  # Download all scenes
  python batch_process_parallel.py --allow_patterns "*.zip"
"""
    )

    # Pattern-based selection (primary method)
    parser.add_argument(
        "--allow_patterns",
        nargs="+",
        type=str,
        default=None,
        help="File patterns to download from HuggingFace (e.g., '00*.zip' '01*.zip')"
    )

    # Manual scene selection (alternative method)
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        help="Specific scene IDs to process (e.g., 0001 0002 0003)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start scene number (use with --end for range)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End scene number (use with --start for range)"
    )

    # Processing options
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)"
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("./work"),
        help="Working directory (default: ./work)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=True,
        help="Cleanup after successful processing (default: True)"
    )
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Do not cleanup after processing (keep downloaded files)"
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Skip upload to R2 (only download and process)"
    )

    args = parser.parse_args()

    # Determine scenes to process
    if args.allow_patterns:
        # Get scenes from HuggingFace matching patterns
        scene_ids = get_scene_files_from_hf(args.allow_patterns)
    elif args.scenes:
        scene_ids = args.scenes
    elif args.start is not None and args.end is not None:
        scene_ids = [f"{i:04d}" for i in range(args.start, args.end + 1)]
    else:
        print("Error: Must specify --allow_patterns, --scenes, or --start/--end")
        print("\nExamples:")
        print('  python batch_process_parallel.py --allow_patterns "00*.zip"')
        print('  python batch_process_parallel.py --scenes 0001 0002 0003')
        print('  python batch_process_parallel.py --start 1 --end 10')
        sys.exit(1)

    if not scene_ids:
        print("No scenes found matching the specified patterns/criteria")
        sys.exit(1)

    cleanup = not args.no_cleanup
    work_dir = args.work_dir.resolve()
    script_dir = Path(__file__).parent.resolve()

    work_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("Parallel Batch Processing AerialMegaDepth Scenes")
    print("=" * 60)
    print(f"Source: HuggingFace ({HF_REPO_ID})")
    print(f"Scenes: {len(scene_ids)} ({scene_ids[0]} - {scene_ids[-1]})")
    print(f"Workers: {args.workers}")
    print(f"Work directory: {work_dir}")
    print(f"Cleanup: {cleanup}")
    print(f"Skip upload: {args.skip_upload}")
    print("=" * 60)
    print()

    start_time = time.time()
    results: list[ProcessResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_scene = {
            executor.submit(
                process_scene,
                scene_id,
                work_dir,
                script_dir,
                cleanup,
                args.skip_upload,
            ): scene_id
            for scene_id in scene_ids
        }

        # Collect results as they complete
        for future in as_completed(future_to_scene):
            result = future.result()
            results.append(result)

    total_duration = time.time() - start_time

    # Summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print()
    print("=" * 60)
    print("Batch Processing Complete")
    print("=" * 60)
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    if results:
        print(f"Average per scene: {total_duration/len(results):.1f}s")
    print()
    print(f"Successful: {len(successful)} scenes")
    if successful:
        print(f"  {' '.join(r.scene_id for r in sorted(successful, key=lambda x: x.scene_id))}")
    print()
    print(f"Failed: {len(failed)} scenes")
    if failed:
        for r in sorted(failed, key=lambda x: x.scene_id):
            print(f"  {r.scene_id}: {r.error[:100] if r.error else 'Unknown error'}...")
    print("=" * 60)

    # Cleanup work directory if empty
    if cleanup and work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
        print(f"Removed empty work directory: {work_dir}")

    # Exit with error code if any failed
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
