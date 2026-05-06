import torch

# --- HSIC utilities ---
def rbf_kernel(x, sigma=None):
    # x: (n, d)
    pairwise_dists = torch.cdist(x, x, p=2) ** 2
    if sigma is None:
        # median heuristic
        sigma = torch.sqrt(torch.median(pairwise_dists))
    K = torch.exp(-pairwise_dists / (2 * sigma ** 2 + 1e-8))
    return K

def laplacian_kernel(x, gamma=None):
    """
    Laplacian kernel: K_ij = exp(-gamma * ||x_i - x_j||)
    If gamma is None, sets gamma = 1/(2*median(distances)).
    """
    dists = torch.cdist(x, x, p=2)
    if gamma is None:
        gamma = 1.0 / (2 * torch.median(dists) + 1e-8)
    return torch.exp(-gamma * dists)

def inverse_multiquadric_kernel(x, beta=0.5, alpha=None):
    """
    Inverse multiquadric kernel: K_ij = (1 + ||x_i - x_j||^2 / alpha^2)^(-beta)
    If alpha is None, uses mean pairwise distance.
    """
    pairwise_dists = torch.cdist(x, x, p=2).pow(2)
    if alpha is None:
        alpha = torch.mean(torch.sqrt(pairwise_dists))
    return (1 + pairwise_dists / (alpha**2 + 1e-8)) ** (-beta)

def cosine_kernel(x, eps=1e-8):
    """
    Cosine similarity kernel: K_ij = (x_i^T x_j) / (||x_i|| * ||x_j||).
    """
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
    return x_norm @ x_norm.T

def center_gram(K):
    n = K.size(0)
    H = torch.eye(n, device=K.device) - 1.0 / n * torch.ones((n, n), device=K.device)
    return H @ K @ H

def hsic(X, Y):
    kernel_function = rbf_kernel
    # kernel_function = inverse_multiquadric_kernel
    # kernel_function = cosine_kernel
    # X, Y: (n, d1), (n, d2)
    Kx = center_gram(kernel_function(X))
    Ky = center_gram(kernel_function(Y))
    n = X.size(0)
    return torch.trace(Kx @ Ky) / ((n - 1) ** 2)

# --- DualHSIC losses ---
def hsic_bottleneck_loss(mu, x, y, lambda_x=1.0, lambda_y=1.0):
    # maximize dependence on labels Y, minimize on inputs X
    return lambda_x * hsic(mu, x) - lambda_y * hsic(mu, y)

def hsic_alignment_loss(model, z_new, z_old, lambda_ha=1.0):
    # minimize drift between new and old representations
    return - 0.5 * lambda_ha * (hsic(z_new, model.proj_head(z_old)) + (hsic(model.proj_head(z_new), z_old)))