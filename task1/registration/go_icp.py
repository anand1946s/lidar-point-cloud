import os
import tempfile
import subprocess
import numpy as np
import open3d as o3d
from pathlib import Path

def write_goicp_cloud(pcd, filepath):
    """Writes an Open3D point cloud to the format expected by Go-ICP (N followed by X Y Z)."""
    pts = np.asarray(pcd.points)
    with open(filepath, 'w') as f:
        f.write(f"{len(pts)}\n")
        np.savetxt(f, pts, fmt='%.6f')

def run_go_icp(source, target, initial_transform, threshold, bin_path, mse_thresh=0.001, trim_fraction=0.0, dt_size=300):
    """
    Runs the compiled Go-ICP C++ executable via subprocess.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Go-ICP executable not found at {bin_path}. Please compile it first.")

    # Apply initial transform
    source_tmp = source.transform(initial_transform)

    # Normalize point clouds to [-1, 1]^3 as required by Go-ICP README
    source_pts = np.asarray(source_tmp.points)
    target_pts = np.asarray(target.points)
    
    all_pts = np.vstack((source_pts, target_pts))
    center = np.mean(all_pts, axis=0)
    
    # Translate to origin
    s_centered = source_pts - center
    t_centered = target_pts - center
    
    # Scale to [-1, 1]
    max_dist = np.max(np.abs(np.vstack((s_centered, t_centered))))
    scale = 1.0 / max_dist if max_dist > 0 else 1.0
    
    s_norm = o3d.geometry.PointCloud()
    s_norm.points = o3d.utility.Vector3dVector(s_centered * scale)
    
    t_norm = o3d.geometry.PointCloud()
    t_norm.points = o3d.utility.Vector3dVector(t_centered * scale)

    # Create temporary directory for I/O
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.txt")
        target_path = os.path.join(tmpdir, "target.txt")
        config_path = os.path.join(tmpdir, "config.txt")
        output_path = os.path.join(tmpdir, "output.txt")

        # Write normalized point clouds
        write_goicp_cloud(s_norm, source_path) # Data
        write_goicp_cloud(t_norm, target_path) # Model

        # Create Go-ICP config file
        config_content = f"""MSEThresh={mse_thresh}
rotMinX=-3.1415926
rotMinY=-3.1415926
rotMinZ=-3.1415926
rotWidth=6.2831853
transMinX=-0.5
transMinY=-0.5
transMinZ=-0.5
transWidth=1.0
trimFraction={trim_fraction}
distTransSize={dt_size}
distTransExpandFactor=2.0
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        # Number of points to use (0 means all points)
        nd_downsampled = 0 

        # Call the Go-ICP executable
        cmd = [
            str(bin_path), 
            target_path, 
            source_path, 
            str(nd_downsampled), 
            config_path, 
            output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        # Read output transformation
        with open(output_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        # Line 0 is time
        # Lines 1,2,3 are the 3x3 Rotation matrix
        # Lines 4,5,6 are the 3x1 Translation vector
        R_goicp = np.array([list(map(float, lines[1].split())),
                            list(map(float, lines[2].split())),
                            list(map(float, lines[3].split()))])
        
        t_goicp = np.array([float(lines[4]), float(lines[5]), float(lines[6])])

    # Un-normalize the transformation matrix back to the original coordinate space
    # t_original = R * s_original - R * c + c + t_goicp / S
    t_original = -np.dot(R_goicp, center) + center + (t_goicp / scale)
    
    T_goicp = np.eye(4)
    T_goicp[:3, :3] = R_goicp
    T_goicp[:3, 3] = t_original

    # Combine with the initial transform
    final_transform = T_goicp @ initial_transform

    # Evaluate using Open3D to get standard metrics (fitness, inlier_rmse) so it behaves identically to run_icp
    result = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, final_transform
    )
    return result