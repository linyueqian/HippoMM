import numpy as np
import torch
from typing import Union, Tuple, List
import scipy.stats as stats

def cosine_similarity(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor]
) -> float:
    """Compute cosine similarity between two vectors"""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
        
    # Flatten if needed
    a = a.reshape(-1)
    b = b.reshape(-1)
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_entropy(features: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute entropy of feature vector"""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
        
    # Normalize to probability distribution
    features = features.reshape(-1)
    features = np.abs(features)
    features = features / np.sum(features)
    
    # Remove zeros to avoid log(0)
    features = features[features > 0]
    
    return -np.sum(features * np.log2(features))

def temporal_overlap(
    t1: Tuple[float, float],
    t2: Tuple[float, float],
    threshold: float = 0.5
) -> bool:
    """Check if two time intervals overlap significantly"""
    start1, end1 = t1
    start2, end2 = t2
    
    overlap = min(end1, end2) - max(start1, start2)
    if overlap <= 0:
        return False
        
    duration1 = end1 - start1
    duration2 = end2 - start2
    
    overlap_ratio = overlap / min(duration1, duration2)
    return overlap_ratio >= threshold

def spatial_distance(
    coord1: Tuple[int, int],
    coord2: Tuple[int, int],
    grid_size: Tuple[int, int] = (16, 16)
) -> float:
    """Compute normalized spatial distance between grid coordinates"""
    x1, y1 = coord1
    x2, y2 = coord2
    
    # Compute Euclidean distance
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Normalize by maximum possible distance in grid
    max_dist = np.sqrt(grid_size[0]**2 + grid_size[1]**2)
    return dist / max_dist

def feature_flow(
    features1: Union[np.ndarray, torch.Tensor],
    features2: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.7
) -> bool:
    """Check if there's a smooth feature flow between two feature vectors"""
    sim = cosine_similarity(features1, features2)
    return sim >= threshold

def merge_features(
    features_list: List[Union[np.ndarray, torch.Tensor]],
    weights: List[float] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Merge multiple feature vectors with optional weights"""
    if weights is None:
        weights = [1.0] * len(features_list)
    
    # Convert all to numpy
    features = []
    for f in features_list:
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()
        features.append(f.reshape(1, -1))
    
    # Weight and combine
    weighted_sum = np.sum([w * f for w, f in zip(weights, features)], axis=0)
    
    # Normalize
    return weighted_sum / np.linalg.norm(weighted_sum)

def gaussian_temporal_weighting(
    times: np.ndarray,
    center: float,
    sigma: float = 1.0
) -> np.ndarray:
    """Compute Gaussian temporal weighting centered at a specific time"""
    return stats.norm.pdf(times, center, sigma)

def compute_feature_statistics(
    features: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, float, float]:
    """Compute basic statistics of feature vector"""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    features = features.reshape(-1)
    return (
        float(np.mean(features)),
        float(np.std(features)),
        float(compute_entropy(features))
    )

def normalize_features(
    features: Union[np.ndarray, torch.Tensor],
    method: str = 'l2'
) -> Union[np.ndarray, torch.Tensor]:
    """Normalize feature vector using specified method"""
    if isinstance(features, torch.Tensor):
        is_torch = True
        features = features.detach().cpu().numpy()
    else:
        is_torch = False
    
    features = features.reshape(-1)
    
    if method == 'l2':
        normalized = features / np.linalg.norm(features)
    elif method == 'l1':
        normalized = features / np.sum(np.abs(features))
    elif method == 'max':
        normalized = features / np.max(np.abs(features))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_torch:
        normalized = torch.from_numpy(normalized)
    
    return normalized

def top_k_cosine_similarity(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Find top k indices of cosine similarities between a single vector and multiple vectors.
    
    Args:
        a: Single feature vector of shape (1024,)
        b: Multiple feature vectors of shape (N, 1024)
        k: Number of top indices to return
        
    Returns:
        Tuple of (top_k_indices, top_k_similarities)
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    
    # Ensure a is 1D and b is 2D
    a = a.reshape(-1)
    if len(b.shape) == 1:
        b = b.reshape(1, -1)
    
    # Compute cosine similarities efficiently using matrix multiplication
    # Normalize vectors first
    a_norm = np.linalg.norm(a)
    b_norms = np.linalg.norm(b, axis=1)
    
    # Compute similarities
    similarities = np.dot(b, a) / (b_norms * a_norm)
    
    # Get top k indices and values
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_similarities = similarities[top_k_indices]
    
    return top_k_indices, top_k_similarities
