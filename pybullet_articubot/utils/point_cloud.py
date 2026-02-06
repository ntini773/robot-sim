"""
Point cloud utilities.
Handles depth to point cloud conversion, FPS sampling, and filtering.
"""

import numpy as np
from typing import Tuple, Optional

def depth_to_point_cloud(depth: np.ndarray, view_matrix: list, projection_matrix: list, 
                         width: int, height: int, far: float = 100.0, near: float = 0.01) -> np.ndarray:
    """
    Convert depth image to point cloud in world coordinates.
    
    Args:
        depth: Depth buffer [H, W]
        view_matrix: View matrix list (16)
        projection_matrix: Projection matrix list (16)
        width: Image width
        height: Image height
        far: Far plane distance (must match projection matrix)
        near: Near plane distance
        
    Returns:
        Point cloud array [N, 3]
    """
    view_matrix = np.array(view_matrix).reshape(4, 4, order='F')
    projection_matrix = np.array(projection_matrix).reshape(4, 4, order='F')
    
    tran_pix_world = np.linalg.inv(np.matmul(projection_matrix, view_matrix))
    
    # Create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2/height, -1:1:2/width]
    y *= -1.  # Invert Y axis for correct image coordinates
    
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = depth.reshape(-1)
    
    # Calculate real Z value from depth buffer
    # PyBullet depth buffer is normalized non-linear buffer
    # stored_depth = far * near / (far - (far - near) * depth_buffer) # This calculation is often implicit or reversed
    
    # Standard OpenGL depth buffer reconstruction
    # The depth buffer in PyBullet `getCameraImage` is already the non-linear buffer value [0, 1]
    # We transform it back to NDC z: z_ndc = 2 * depth - 1
    z_ndc = 2 * z - 1
    
    # Stack pixels [x_ndc, y_ndc, z_ndc, 1]
    pixels = np.stack([x, y, z_ndc, np.ones_like(z)], axis=1)
    
    # Filter out far plane points (background)
    mask = z < 0.999 # Typical max depth is 1.0
    pixels = pixels[mask]
    
    # Transform pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    
    # Perspective division
    points /= points[:, 3:4]
    
    return points[:, :3]

def fps_downsample(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) to select a fixed number of points.
    
    Args:
        points: Input point cloud [N, 3]
        num_points: Target number of points
        
    Returns:
        Downsampled point cloud [num_points, 3]
    """
    N, D = points.shape
    if N < num_points:
        # If fewer points than requested, pad with repetition
        indices = np.random.choice(N, num_points, replace=True)
        return points[indices]
    
    # Basic FPS implementation
    # Initialize with random point
    xyz = points
    centroids = np.zeros((num_points,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_points):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        
    return points[centroids.astype(int)]

def filter_by_segmentation(points: np.ndarray, segmentation: np.ndarray, 
                           depth: np.ndarray, object_id: int) -> np.ndarray:
    """
    Filter points belonging to a specific object using segmentation mask.
    Note: 'points' input here assumes it corresponds one-to-one with depth/segmentation pixels.
    If 'points' is already unstructured, this function won't work directly.
    
    Alternative approach: pass raw maps and convert only masked pixels.
    """
    # This logic matches depth_to_point_cloud structure, assuming we haven't flattened yet
    # Or, we can modify depth_to_point_cloud to accept mask.
    pass
    
def get_object_point_cloud(depth: np.ndarray, segmentation: np.ndarray, 
                           view_matrix: list, projection_matrix: list, 
                           width: int, height: int, object_id: int) -> np.ndarray:
    """
    Extract point cloud only for a specific object.
    """
    mask = (segmentation == object_id)
    if not np.any(mask):
        return np.zeros((0, 3))
    
    # Create depth map with only object (set others to far plane)
    masked_depth = depth.copy()
    # Mask out non-object pixels (set to 1.0 which is far plane)
    masked_depth[~mask] = 1.0 
    
    points = depth_to_point_cloud(masked_depth, view_matrix, projection_matrix, width, height)
    return points
