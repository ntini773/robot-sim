import rerun as rr
import open3d as o3d
import pathlib
import numpy as np
import argparse
import re
import cv2

def main():
    parser = argparse.ArgumentParser(description="Visualize PLY point clouds, depth, and RGB in Rerun.")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing .ply files")
    parser.add_argument("--depth_dir", type=str, default=None, help="Directory containing .npy depth files")
    parser.add_argument("--rgb_dir", type=str, default=None, help="Directory containing .png RGB files")
    parser.add_argument("--save", type=str, default=None, help="Save the recording to a .rrd file")
    args = parser.parse_args()

    if not args.dir and not args.depth_dir and not args.rgb_dir:
        print("Error: At least one of --dir, --depth_dir, or --rgb_dir must be specified.")
        return

    # Initialize Rerun
    rr.init("multimodal_robot_data", spawn=False)

    if args.save:
        print(f"Saving recording to {args.save}")
        rr.save(args.save)
    else:
        print("Spawning Rerun viewer...")
        rr.spawn()

    # Log world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

    # Collect files
    ply_files = []
    if args.dir:
        data_dir = pathlib.Path(args.dir)
        if not data_dir.exists():
            print(f"Directory {data_dir} does not exist.")
            return
        ply_files = sorted(list(data_dir.glob("*.ply")), key=natural_sort_key)
        print(f"Found {len(ply_files)} PLY files.")

    depth_files = []
    if args.depth_dir:
        depth_path = pathlib.Path(args.depth_dir)
        if not depth_path.exists():
            print(f"Directory {depth_path} does not exist.")
            return
        depth_files = sorted(list(depth_path.glob("*.npy")), key=natural_sort_key)
        print(f"Found {len(depth_files)} depth files.")

    rgb_files = []
    if args.rgb_dir:
        rgb_path = pathlib.Path(args.rgb_dir)
        if not rgb_path.exists():
            print(f"Directory {rgb_path} does not exist.")
            return
        rgb_files = sorted(list(rgb_path.glob("*.png")), key=natural_sort_key)
        print(f"Found {len(rgb_files)} RGB files.")

    num_frames = max(len(ply_files), len(depth_files), len(rgb_files))

    if num_frames == 0:
        print("No files found in any specified directory.")
        return

    print(f"Starting visualization for {num_frames} frames...")

    for i in range(num_frames):
        rr.set_time("frame", sequence=i)

        # Log Point Cloud
        if i < len(ply_files):
            pcd = o3d.io.read_point_cloud(str(ply_files[i]))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            if colors.size > 0:
                rr.log("world/point_cloud", rr.Points3D(positions=points, colors=colors))
            else:
                rr.log("world/point_cloud", rr.Points3D(positions=points))

        # Log Depth
        if i < len(depth_files):
            depth_data = np.load(depth_files[i])
            rr.log("camera/depth", rr.DepthImage(depth_data, meter=1.0))

        # Log RGB
        if i < len(rgb_files):
            rgb_data = cv2.imread(str(rgb_files[i]))
            rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
            rr.log("camera/rgb", rr.Image(rgb_data))

        if i % 20 == 0:
            print(f"Processed frame {i}/{num_frames}")

    print("Done logging.")

if __name__ == "__main__":
    main()