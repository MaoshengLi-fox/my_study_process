# 把你想暴露的函数名对齐到简洁的 API
from .path_helper import(
  get_data_path
)

from .dimensionality_reduction import (
    pca_reduce,
    pca_inverse_transform,
    # lda_reduce,
    # umap_reduce,
)

__all__ = ["pca_reduce","pca_inverse_transform", "lda_reduce", "umap_reduce","get_data_path"]
