import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# ---- 1. 准备数据：X 形状 (n_samples, n_features) ----
# 这里用随机数据举例；你的真实数据替换 X 即可
rng = np.random.default_rng(0)
n, p = 500, 12
X = rng.normal(size=(n, p))

# 一般先标准化：使各特征均值0方差1
X_std = StandardScaler().fit_transform(X)

# ---- 2. 选择因子数：AIC/BIC/对数似然  ----
def fa_ic(Xz, k_list, criterion="bic", max_iter=200):
    """
    使用 sklearn 的 FactorAnalysis.score_（平均对数似然）来计算 AIC/BIC
    AIC = -2*LL + 2*params
    BIC = -2*LL + params*log(n)
    params 近似 = p*k + p（载荷Λ + 独特方差Ψ）
    """
    n, p = Xz.shape
    out = []
    for k in k_list:
        fa = FactorAnalysis(n_components=k, max_iter=max_iter)
        fa.fit(Xz)
        ll = fa.score(Xz) * n           # score_ 是平均对数似然
        params = p*k + p
        aic = -2*ll + 2*params
        bic = -2*ll + params*np.log(n)
        out.append((k, ll, aic, bic))
    return out

cand_ks = range(1, min(p, 10)+1)
sel = fa_ic(X_std, cand_ks, criterion="bic")
best_k = sorted(sel, key=lambda t: t[3])[0][0]   # 按 BIC 最小选
print("BIC 选择的因子数 best_k =", best_k)

# ---- 3. 训练最终模型 ----
fa = FactorAnalysis(n_components=best_k, rotation=None, max_iter=500)
fa.fit(X_std)

# 因子载荷（p × k）：每一列是一个因子在各变量上的载荷
loadings = fa.components_.T   # 注意 sklearn 的components_ 是 (k × p)，转置成 (p × k)
# 因子得分（n × k）：各样本在每个因子上的分数
scores = fa.transform(X_std)

# 共性度（communality）与特异度（uniqueness）
# FA 模型：Σ ≈ ΛΛ^T + Ψ，其中 Ψ 为对角矩阵（特异方差）
communality = np.sum(loadings**2, axis=1)        # 每个变量的公共方差贡献
uniqueness = fa.noise_variance_                  # Ψ 的对角
print("共性度（前5个变量）:", communality[:5])
print("特异度（前5个变量）:", uniqueness[:5])

# ---- 4. Varimax 正交旋转（让载荷更“稀疏”、便于解释）----
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """
    Varimax 旋转：输入 (p × k) 的载荷矩阵，输出旋转后的载荷与旋转矩阵
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0)))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and (d - d_old) < tol:
            break
    return Phi @ R, R

rot_loadings, R = varimax(loadings)
# 如果想要对应的旋转后因子得分（正交旋转下）：
rot_scores = scores @ R

# ---- 5. 打印前几项结果便于检查 ----
np.set_printoptions(precision=3, suppress=True)
print("\n原始载荷 (前5行):\n", loadings[:5])
print("\nVarimax后载荷 (前5行):\n", rot_loadings[:5])
print("\n样本因子得分 (前5行):\n", scores[:5])
print("\nVarimax后因子得分 (前5行):\n", rot_scores[:5])
