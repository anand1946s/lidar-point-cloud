import open3d as o3d
import numpy as np
import time

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

def main():
    # 1. Update this path to where your bun090.ply file is located!
    ply_path = "bunny/bunny/data/bun090.ply" 
    
    print(f"Loading local point cloud file: {ply_path} ...")
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
        
    if not pcd.has_points():
        print("Point cloud is empty. Please verify the file path.")
        return

    # Base cloud cleanup and sizing metrics
    pcd.paint_uniform_color([0.6, 0.6, 0.6]) # Matte gray background
    
    # Calculate bounding box to sanity check scale sizes
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"Point cloud loaded with {len(pcd.points)} points.")
    print(f"Bounding box extents (X, Y, Z): {bbox.get_extent()}")

    # 2. Run custom SIFT 
    # Adjust sigma depending on the spatial bounds shown by the print statement above
    start = time.time()
    keypoints = extract_3d_sift_keypoints(pcd, sigma=0.004, k=1.414, threshold=0.0002)
    print(f"\n3D-SIFT completed in {time.time() - start:.2f} seconds.")
    print(f"Extracted {len(keypoints.points)} SIFT keypoints.")

    # 3. Create visual marker spheres for keypoints
    visual_keypoints = o3d.geometry.TriangleMesh()
    for pt in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0015)
        sphere.translate(pt)
        visual_keypoints += sphere
    visual_keypoints.paint_uniform_color([1.0, 0.0, 0.0]) # Red spheres

    # 4. Render visualization window
    print("Opening visualizer...")
    o3d.visualization.draw_geometries([pcd, visual_keypoints], 
                                      window_name="Custom 3D-SIFT on bun090.ply")

if __name__ == "__main__":
    main()