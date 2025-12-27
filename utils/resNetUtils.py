import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def matrix_to_quaternion(matrix_3x3):
    """Converte una matrice di rotazione 3x3 in quaternione (x, y, z, w)."""
    r = R.from_matrix(matrix_3x3)
    return torch.tensor(r.as_quat(), dtype=torch.float32)

def rotation_loss(q_pred, q_true):
    """Calcola la loss basata sul prodotto scalare assoluto[cite: 110]."""
    # Normalizzazione dei quaternioni
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    q_true = q_true / torch.norm(q_true, dim=1, keepdim=True)
    
    # Formula: 1 - |q_pred · q_true| [cite: 110]
    inner_prod = torch.abs(torch.sum(q_pred * q_true, dim=1))
    return torch.mean(1 - inner_prod)

def compute_pinhole_translation(bbox, intrinsics, real_diameter):
    """Calcola la traslazione geometrica 2D -> 3D."""
    x, y, w, h = bbox
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # Stima Z (profondità) usando il diametro reale dell'oggetto [cite: 68]
    pixel_size = max(w, h)
    Z = (fx * real_diameter) / pixel_size

    # Inverse Projection per X e Y [cite: 38]
    u_center = x + w / 2
    v_center = y + h / 2
    X = (u_center - cx) * Z / fx
    Y = (v_center - cy) * Z / fy

    return torch.tensor([X, Y, Z])