import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def compute_psi(ref, cur, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between reference and current distributions.

    规则：
      - 仅当 ref 与 cur 两侧 **同时为数值型** 时，走“数值 PSI”（用 ref 的分位数分箱）
      - 否则，一律走“分类 PSI”（基于类别频率）

    这样可以避免任意一侧是字符串/object/string-dtype 时落入数值分支导致 astype(float) 报错。
    """
    ref_s = pd.Series(ref).dropna()
    cur_s = pd.Series(cur).dropna()

    # 只有两侧都为数值 dtype 才走数值分支
    both_numeric = ptypes.is_numeric_dtype(ref_s) and ptypes.is_numeric_dtype(cur_s)

    # ============================ 分类 PSI（默认分支） ============================
    if not both_numeric:
        # 统一转为 object 做频率统计，兼容 object / string dtype / category / 混合类型
        ref_o = ref_s.astype("object")
        cur_o = cur_s.astype("object")

        cats = pd.Index(ref_o.unique()).union(pd.Index(cur_o.unique()))

        ref_counts = ref_o.value_counts().reindex(cats, fill_value=0).astype(float)
        cur_counts = cur_o.value_counts().reindex(cats, fill_value=0).astype(float)

        # 归一化（若为空则退化为均匀分布，避免 0/0）
        ref_sum = ref_counts.sum()
        cur_sum = cur_counts.sum()
        ref_dist = (
            (ref_counts / ref_sum) if ref_sum > 0 else np.ones(len(cats)) / len(cats)
        )
        cur_dist = (
            (cur_counts / cur_sum) if cur_sum > 0 else np.ones(len(cats)) / len(cats)
        )

        # 平滑，避免 log(0)
        eps = 1e-8
        ref_dist = np.where(ref_dist == 0, eps, ref_dist)
        cur_dist = np.where(cur_dist == 0, eps, cur_dist)

        return float(np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist)))

    # ============================ 数值 PSI ============================
    # 两侧确为数值时才会到这里
    ref_f = ref_s.astype(float)
    cur_f = cur_s.astype(float)

    # 用参考样本分位点分箱，保证跨期可比
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(ref_f.quantile(quantiles).values)

    # 参考样本无方差/分箱无效：定义 PSI=0
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_f, bins=breakpoints)
    cur_counts, _ = np.histogram(cur_f, bins=breakpoints)

    ref_dist = ref_counts / max(len(ref_f), 1)
    cur_dist = cur_counts / max(len(cur_f), 1)

    eps = 1e-8
    ref_dist = np.where(ref_dist == 0, eps, ref_dist)
    cur_dist = np.where(cur_dist == 0, eps, cur_dist)

    return float(np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist)))
