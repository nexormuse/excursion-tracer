"""방법 간 쌍대 비교: 같은 시나리오 쌍으로 McNemar 검정."""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_paired(a: pd.Series, b: pd.Series) -> dict:
    """a, b: 같은 시나리오 인덱스의 적중 여부. 정확 McNemar (불일치 쌍의 이항검정)."""
    df = pd.concat({"a": a, "b": b}, axis=1).dropna().astype(bool)
    n11 = int((df["a"] & df["b"]).sum())
    n10 = int((df["a"] & ~df["b"]).sum())
    n01 = int((~df["a"] & df["b"]).sum())
    n00 = int((~df["a"] & ~df["b"]).sum())
    p = float(mcnemar([[n11, n10], [n01, n00]], exact=True).pvalue) if n10 + n01 else 1.0
    return {"n": len(df), "both": n11, "only_a": n10, "only_b": n01, "neither": n00, "p": p}
