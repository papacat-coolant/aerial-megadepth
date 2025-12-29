#!/usr/bin/env python3
"""
Convert AerialMegaDepth COLMAP output to unified format for nerfvis visualization.

Input (COLMAP format):
    - sparse-txt/cameras.txt: Camera intrinsics
    - sparse-txt/images.txt: Camera poses (quaternion + translation)
    - depths/*.h5: Depth maps in HDF5 format
    - images/*.jpeg: RGB images

Output (unified format):
    - image_names.json: List of image names
    - cam_from_worlds.npy: (N, 3, 4) w2c matrices
    - intrinsics.npy: (N, 3, 3) intrinsics
    - depths/*.npy: Depth maps
    - images/*.jpg: RGB images (symlinked or copied)

Usage:
    python convert_to_unified.py --input_dir data/0001/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric --output_dir output_unified
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    # scipy uses (x, y, z, w) format
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_matrix()


def parse_cameras_txt(cameras_path):
    """Parse COLMAP cameras.txt file.

    Returns:
        dict: camera_id -> (width, height, fx, fy, cx, cy)
    """
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            if model == 'SIMPLE_PINHOLE':
                # SIMPLE_PINHOLE: f, cx, cy
                f = float(parts[4])
                cx = float(parts[5])
                cy = float(parts[6])
                fx = fy = f
            elif model == 'PINHOLE':
                # PINHOLE: fx, fy, cx, cy
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])
            else:
                print(f"Warning: Unsupported camera model {model}, using SIMPLE_PINHOLE fallback")
                f = float(parts[4])
                cx = float(parts[5]) if len(parts) > 5 else width / 2
                cy = float(parts[6]) if len(parts) > 6 else height / 2
                fx = fy = f

            cameras[camera_id] = (width, height, fx, fy, cx, cy)

    return cameras


def parse_images_txt(images_path):
    """Parse COLMAP images.txt file.

    Returns:
        list of tuples: (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)
    """
    images = []
    with open(images_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]

            images.append((image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name))

            # Skip the next line (POINTS2D data)
            i += 2
        else:
            i += 1

    return images


def load_h5_depth(h5_path):
    """Load depth map from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Common keys for depth data
        for key in ['depth', 'dataset', 'data']:
            if key in f:
                return np.array(f[key])
        # If no common key found, use the first dataset
        keys = list(f.keys())
        if keys:
            return np.array(f[keys[0]])
    return None


def convert_to_unified(input_dir, output_dir, max_images=None, copy_images=False):
    """Convert COLMAP format to unified format.

    Args:
        input_dir: Path to COLMAP output directory (localized_dense_metric)
        output_dir: Path to output unified format directory
        max_images: Maximum number of images to convert (None for all)
        copy_images: If True, copy images; if False, create symlinks
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'depths').mkdir(exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)

    # Parse COLMAP files
    cameras_path = input_dir / 'sparse-txt' / 'cameras.txt'
    images_path = input_dir / 'sparse-txt' / 'images.txt'

    if not cameras_path.exists():
        cameras_path = input_dir / 'sparse' / 'cameras.txt'
    if not images_path.exists():
        images_path = input_dir / 'sparse' / 'images.txt'

    print(f"Parsing cameras from: {cameras_path}")
    cameras = parse_cameras_txt(cameras_path)
    print(f"  Found {len(cameras)} cameras")

    print(f"Parsing images from: {images_path}")
    images = parse_images_txt(images_path)
    print(f"  Found {len(images)} images")

    if max_images:
        images = images[:max_images]
        print(f"  Limiting to {max_images} images")

    # Prepare output arrays
    image_names = []
    cam_from_worlds = []
    intrinsics = []

    # Process each image
    for idx, (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name) in enumerate(images):
        # Get camera intrinsics
        if camera_id not in cameras:
            print(f"  Warning: Camera {camera_id} not found for image {name}, skipping")
            continue

        width, height, fx, fy, cx, cy = cameras[camera_id]

        # Build intrinsics matrix (3x3)
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # Build w2c (world-to-camera) matrix
        # COLMAP stores: R @ X_world + t = X_cam
        # So w2c = [R | t]
        R = quat_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])

        w2c = np.zeros((3, 4))
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        # Check if depth file exists
        # Try different naming conventions
        depth_name = None

        # Handle different naming patterns:
        # - Aerial: 0001_XXX.jpeg -> 0001_XXX.h5
        # - MegaDepth: XXXXX_o.jpg.jpg -> XXXXX_o.jpg.h5
        if name.endswith('.jpg.jpg'):
            # MegaDepth format: remove last .jpg and replace with .h5
            base = name[:-4]  # Remove .jpg
            possible_depth_names = [base + '.h5']
        elif name.endswith('.jpeg'):
            # Aerial format
            base = name[:-5]  # Remove .jpeg
            possible_depth_names = [base + '.h5']
        else:
            # Generic fallback
            possible_depth_names = [
                name.replace('.jpeg', '.h5').replace('.jpg', '.h5'),
                name + '.h5',
            ]

        for dn in possible_depth_names:
            if (input_dir / 'depths' / dn).exists():
                depth_name = dn
                break

        if depth_name is None:
            print(f"  Warning: Depth not found for {name}, skipping")
            continue

        # Check if image exists
        src_img_path = input_dir / 'images' / name
        if not src_img_path.exists():
            print(f"  Warning: Image not found: {src_img_path}, skipping")
            continue

        # Load and save depth
        depth_path = input_dir / 'depths' / depth_name
        depth = load_h5_depth(depth_path)
        if depth is None:
            print(f"  Warning: Failed to load depth from {depth_path}, skipping")
            continue

        # Save depth as .npy
        # Normalize output names
        if name.endswith('.jpg.jpg'):
            out_base = name[:-8]  # Remove .jpg.jpg
        elif name.endswith('.jpeg'):
            out_base = name[:-5]  # Remove .jpeg
        elif name.endswith('.jpg'):
            out_base = name[:-4]  # Remove .jpg
        else:
            out_base = name

        out_depth_name = out_base + '.npy'
        np.save(output_dir / 'depths' / out_depth_name, depth)

        # Copy or symlink image
        out_img_name = out_base + '.jpg'
        dst_img_path = output_dir / 'images' / out_img_name

        if copy_images:
            shutil.copy2(src_img_path, dst_img_path)
        else:
            # Create relative symlink
            if dst_img_path.exists() or dst_img_path.is_symlink():
                dst_img_path.unlink()
            rel_path = os.path.relpath(src_img_path, dst_img_path.parent)
            dst_img_path.symlink_to(rel_path)

        # Add to output lists
        image_names.append(out_img_name)
        cam_from_worlds.append(w2c)
        intrinsics.append(K)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(images)} images")

    # Save output files
    print(f"\nSaving {len(image_names)} images to unified format...")

    # Save image names
    with open(output_dir / 'image_names.json', 'w') as f:
        json.dump(image_names, f, indent=2)

    # Save camera matrices
    cam_from_worlds = np.array(cam_from_worlds)
    np.save(output_dir / 'cam_from_worlds.npy', cam_from_worlds)
    print(f"  cam_from_worlds.npy: {cam_from_worlds.shape}")

    # Save intrinsics
    intrinsics = np.array(intrinsics)
    np.save(output_dir / 'intrinsics.npy', intrinsics)
    print(f"  intrinsics.npy: {intrinsics.shape}")

    print(f"\nConversion complete! Output saved to: {output_dir}")
    print(f"  - {len(image_names)} images")
    print(f"  - {len(list((output_dir / 'depths').glob('*.npy')))} depth maps")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert AerialMegaDepth COLMAP output to unified format for visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to COLMAP output directory (localized_dense_metric)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output_unified"),
        help="Path to output unified format directory (default: output_unified)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to convert (default: all)"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images instead of creating symlinks"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    convert_to_unified(
        args.input_dir,
        args.output_dir,
        max_images=args.max_images,
        copy_images=args.copy_images
    )


if __name__ == "__main__":
    main()
