import open3d as o3d
import numpy as np
import time
import os

def compute_gaussian_blur(pcd, pcd_tree, sigma):
    """
    Simulates a 3D Gaussian blur on the Z-coordinate (height) 
    field to track geometric scale-space changes.
    """
    points = np.asarray(pcd.points)
    values = points[:, 2]  # Using Z-axis as our geometry descriptor
    blurred_values = np.zeros_like(values)
    
    search_radius = 3.0 * sigma
    
    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(points[i], search_radius)
        
        if len(idx) == 0:
            blurred_values[i] = values[i]
            continue
            
        neighbor_points = points[idx]
        neighbor_values = values[idx]
        
        dists_sq = np.sum((neighbor_points - points[i]) ** 2, axis=1)
        
        weights = np.exp(-dists_sq / (2 * (sigma ** 2)))
        weights /= np.sum(weights)
        
        blurred_values[i] = np.sum(neighbor_values * weights)
        
    return blurred_values

def extract_3d_sift_keypoints(pcd, sigma=0.003, k=1.414, threshold=0.0005):
    """
    Custom 3D-SIFT Keypoint Detector implementation using k-d trees.
    """
    print("Building k-d tree and calculating scale spaces...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    
    sigma1 = sigma
    sigma2 = sigma * k
    sigma3 = sigma * (k ** 2)
    
    # Scale Space Construction
    L1 = compute_gaussian_blur(pcd, pcd_tree, sigma1)
    L2 = compute_gaussian_blur(pcd, pcd_tree, sigma2)
    L3 = compute_gaussian_blur(pcd, pcd_tree, sigma3)
    
    # Difference of Gaussians (DoG)
    DoG1 = L2 - L1
    DoG2 = L3 - L2
    
    keypoint_indices = []
    
    print("Searching for local DoG extrema across spatial and scale layers...")
    for i in range(len(points)):
        val = DoG1[i]
        
        if abs(val) < threshold:
            continue
            
        # Get 10 nearest spatial neighbors to evaluate local extrema
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], knn=10)
        
        is_max = True
        is_min = True
        
        for n_idx in idx:
            if n_idx == i:
                continue
                
            if DoG1[n_idx] >= val: is_max = False
            if DoG1[n_idx] <= val: is_min = False
            
            if DoG2[n_idx] >= val: is_max = False
            if DoG2[n_idx] <= val: is_min = False
            
        if is_max or is_min:
            keypoint_indices.append(i)
            
    return pcd.select_by_index(keypoint_indices)

def process_ply_file(input_path, output_dir):

    filename = os.path.basename(input_path)

    print(f"\nProcessing {filename}")

    pcd = o3d.io.read_point_cloud(input_path)

    if not pcd.has_points():
        print("Empty point cloud")
        return

    start = time.time()

    keypoints = extract_3d_sift_keypoints(
        pcd,
        sigma=0.004,
        k=1.414,
        threshold=0.0002
    )

    print(
        f"Extracted {len(keypoints.points)} keypoints "
        f"in {time.time() - start:.2f}s"
    )

    output_name = filename.replace(
        ".ply",
        "_keypoints.ply"
    )

    output_path = os.path.join(
        output_dir,
        output_name
    )

    o3d.io.write_point_cloud(
        output_path,
        keypoints
    )

    print(f"Saved: {output_path}")

def main():

    data_dir = "bunny/bunny/data"

    output_dir = "keypoint_outputs"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    bunny_files = [
        "bun000.ply",
        "bun045.ply",
        "bun090.ply",
        "bun180.ply",
        "bun270.ply",
        "bun315.ply",
        "chin.ply",
        "ear_back.ply"
    ]

    for filename in bunny_files:

        input_path = os.path.join(
            data_dir,
            filename
        )

        process_ply_file(
            input_path,
            output_dir
        )

    print("\nDone!")

if __name__ == "__main__":
    main()