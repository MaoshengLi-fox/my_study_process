import numpy as np
import umap

def pca_reduce(data, k=None, whiten=False, center=True, return_model=False):
    """
    data: (n_samples, n_features)
    k: target dimension,None keep all singular value奇异值
    """
    data = np.asarray(data, dtype=float)
    mu = data.mean(axis=0) if center else np.zeros(data.shape[1])
    data_c = data - mu

    # SVD
    U, S, Vt = np.linalg.svd(data_c, full_matrices=False)  # data_c = U @ diag(S) @ Vt
    V = Vt.T
    n = data.shape[0]
    # explained variation and percentage 解释方差与占比
    eigvals = (S**2) / (n - 1)
    total_var = eigvals.sum()
    if k is None:
        k = V.shape[1]

    Vk = V[:, :k]
    Sk = S[:k]
    eigvals_k = eigvals[:k]
    exp_var_ratio = eigvals_k / total_var

    # 低维表示
    Z = data_c @ Vk  # 等价于 U[:, :k] @ np.diag(Sk)

    if whiten:
        Z = Z / np.sqrt(eigvals_k)  # 列级除以对应 sqrt(lambda_i)

    if return_model:
        return {
            "Z": Z,
            "components_": Vk.T,               # 与 sklearn 对齐: (k, d)
            "mean_": mu,
            "explained_variance_": eigvals_k,
            "explained_variance_ratio_": exp_var_ratio,
            "singular_values_": Sk,
            "total_variance_": total_var,
        }
    return Z

# 逆变换重构
def pca_inverse_transform(Z, model):
    Vk = model["components_"].T
    mu = model["mean_"]
    return Z @ Vk.T + mu


def umap_reduce(data,arg):
    reducer = umap.UMAP(n_neighbors=arg.n_neighbors, min_dist=arg.min_dist, n_components=arg.n_components,metric=arg.metric, random_state=42)
    emb = reducer.fit_transform(data)  # -> (N,2)
    return emb