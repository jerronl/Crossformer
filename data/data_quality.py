import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# 检查结果容器
# ─────────────────────────────────────────────
class QualityReport:
    def __init__(self, source: str):
        self.source = source
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str):
        self.errors.append(f"[ERROR] {msg}")

    def warn(self, msg: str):
        self.warnings.append(f"[WARN]  {msg}")

    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"数据质量报告 — {self.source}",
            f"行数: {self._rows}  列数: {self._cols}",
            f"{'='*60}",
        ]
        lines += self.errors + self.warnings
        status = "✅ 通过" if self.ok() else "❌ 存在错误"
        lines.append(
            f"\n{status}（{len(self.errors)} 错误 / {len(self.warnings)} 警告）"
        )
        return "\n".join(lines)

    def set_shape(self, rows, cols):
        self._rows, self._cols = rows, cols


# ─────────────────────────────────────────────
# 核心检查函数
# ─────────────────────────────────────────────
def check_data_quality(
    df: pd.DataFrame,
    source: str = "unknown",
    dtm0: Optional[str] = None,
    required_cols: Optional[list[str]] = None,
    vol_cols: Optional[list[str]] = None,
    price_cols: Optional[list[str]] = None,
    nan_sentinel: float = -9999900,
    raise_on_error: bool = False,
    verbose: bool = True,
) -> QualityReport:
    """
    每次 pd.read_csv 后调用的数据质量检查。

    Parameters
    ----------
    df             : 刚读取的 DataFrame（替换哨兵值之前或之后均可）
    source         : 文件名，用于日志标识
    dtm0           : 主 DTM 列名（如 'dtm_0'），检查非空
    required_cols  : 必须存在的列名列表
    vol_cols       : 隐含波动率列，需在 (0, 5] 合理区间
    price_cols     : 价格列（close/hi/lo/spot），需 > 0
    nan_sentinel   : 哨兵缺失值（默认 -9999900），用于统计残留量
    raise_on_error : 若 True，发现 ERROR 则抛出 ValueError
    """
    report = QualityReport(source)
    report.set_shape(*df.shape)

    # 1. 空表检查
    if df.empty:
        report.error("DataFrame 为空，没有任何行")
        _finalize(report, raise_on_error)
        return report

    if len(df.columns) == 0:
        report.error("DataFrame 没有任何列")
        _finalize(report, raise_on_error)
        return report

    # 2. 必要列存在性
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            report.error(f"缺少必要列: {missing}")

    # 3. 哨兵值残留（未被 replace 的 -9999900）
    numeric_df = df.select_dtypes(include=[np.number])
    sentinel_mask = numeric_df == nan_sentinel
    sentinel_count = sentinel_mask.sum().sum()
    if sentinel_count > 0:
        cols_with_sentinel = sentinel_mask.columns[sentinel_mask.any()].tolist()
        report.warn(
            f"发现 {sentinel_count} 个哨兵值 ({nan_sentinel}) 未被替换，"
            f"涉及列: {cols_with_sentinel}"
        )

    # 4. NaN 检查（替换哨兵后的真实缺失）
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        total_nan = nan_cols.sum()
        nan_ratio = total_nan / (df.shape[0] * df.shape[1])
        if nan_ratio > 0.20:
            report.error(f"NaN 占比过高: {nan_ratio:.1%}，涉及列: {nan_cols.to_dict()}")
        else:
            report.warn(f"存在 NaN: {nan_cols.to_dict()}")

    # 5. 主 DTM 列不能全为空
    if dtm0 and dtm0 in df.columns:
        dtm0_nan = df[dtm0].isna().sum()
        if dtm0_nan == len(df):
            report.error(f"主 DTM 列 '{dtm0}' 全为 NaN，数据无法使用")
        elif dtm0_nan > 0:
            report.warn(f"主 DTM 列 '{dtm0}' 有 {dtm0_nan} 行为 NaN（将被过滤）")
    elif dtm0:
        report.error(f"主 DTM 列 '{dtm0}' 不存在于 DataFrame 中")

    # 6. date 列格式检查
    if "date" in df.columns:
        sample = df["date"].dropna().head(5).astype(str)
        # 期望格式类似 "#2023-01-01 09:00:00" 或 "2023-01-01 09:00:00"
        bad = sample[
            ~sample.str.replace("#", "", regex=False).str.match(r"\d{4}-\d{2}-\d{2}")
        ]
        if not bad.empty:
            report.warn(f"'date' 列存在格式异常的样本: {bad.tolist()}")

    # 7. 隐含波动率列合理性（期望范围 0 < vol <= 5，即 0%~500%）
    if vol_cols:
        for col in vol_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                report.warn(f"波动率列 '{col}' 全为非数值或 NaN")
                continue
            n_neg = (series <= 0).sum()
            n_huge = (series > 5).sum()
            if n_neg > 0:
                report.error(f"波动率列 '{col}' 有 {n_neg} 个 ≤ 0 的值（不合理）")
            if n_huge > 0:
                report.warn(f"波动率列 '{col}' 有 {n_huge} 个 > 500% 的极端值")

    # 8. 价格列合理性（必须 > 0）
    # if price_cols:
    #     for col in price_cols:
    #         if col not in df.columns:
    #             continue
    #         series = pd.to_numeric(df[col], errors="coerce").dropna()
    #         if series.empty:
    #             continue
    #         n_neg = (series <= 0).sum()
    #         if n_neg > 0:
    #             report.error(f"价格列 '{col}' 有 {n_neg} 个 ≤ 0 的值（不合理）")

    # 9. hi/lo 逻辑关系（若同时存在）
    _check_hi_lo(df, report)

    # 10. 重复行检查
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        report.warn(f"存在 {dup_count} 行完全重复")

    # 11. 最小行数保障（少于 10 行视为可疑）
    if len(df) < 10:
        report.warn(f"数据行数过少（{len(df)} 行），可能导致训练/分割失败")

    _finalize(report, raise_on_error, verbose, df=df, dtm0=dtm0)
    return report


def _zscore_outlier_report(
    df: pd.DataFrame, dtm0: Optional[str], top_n: int = 5
) -> str:
    """对所有数值列计算 z-score，每列取 |z| > 3 的前 top_n 行（按 |z| 降序）。"""
    from scipy import stats as sp_stats

    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return ""

    # 识别 id 列用于展示
    id_cols = [c for c in ["ticker", "date", "horizon", dtm0] if c and c in df.columns]

    z_scores = num_df.apply(lambda col: np.abs(sp_stats.zscore(col, nan_policy="omit")))

    records = []
    for col in z_scores.columns:
        outlier_idx = z_scores.index[z_scores[col] > 3]
        if outlier_idx.empty:
            continue
        top = z_scores.loc[outlier_idx, col].nlargest(top_n)
        for idx, z in top.items():
            row = {c: df.at[idx, c] for c in id_cols}
            row["outlier_col"] = col
            row["value"] = df.at[idx, col]
            row["z_score"] = round(z, 2)
            records.append(row)

    if not records:
        return "  （无 |z| > 3 的异常值）"

    result = pd.DataFrame(records).sort_values("z_score", ascending=False)
    return result.to_string(index=False)


def _check_hi_lo(df: pd.DataFrame, report: QualityReport):
    """检查 hi >= lo，以及 close 在 [lo, hi] 范围内"""
    hi_col = next((c for c in ["hi", "high"] if c in df.columns), None)
    lo_col = next((c for c in ["lo", "low"] if c in df.columns), None)
    cl_col = next((c for c in ["close", "cl"] if c in df.columns), None)

    if hi_col and lo_col:
        hi = pd.to_numeric(df[hi_col], errors="coerce")
        lo = pd.to_numeric(df[lo_col], errors="coerce")
        bad = (hi < lo).sum()
        if bad > 0:
            report.error(f"'{hi_col}' < '{lo_col}' 出现 {bad} 次（hi/lo 逻辑矛盾）")

        if cl_col:
            cl = pd.to_numeric(df[cl_col], errors="coerce")
            out_of_range = ((cl < lo) | (cl > hi)).sum()
            if out_of_range > 0:
                report.warn(f"'{cl_col}' 有 {out_of_range} 行不在 [lo, hi] 范围内")


def _finalize(
    report: QualityReport,
    raise_on_error: bool,
    verbose: bool = True,
    df: Optional[pd.DataFrame] = None,
    dtm0: Optional[str] = None,
):
    if verbose:
        print(report.summary())
        if df is not None:
            print(f"\n{'─'*60}")
            print("Z-score 异常值报告（|z| > 3，每列 Top-5，按 |z| 降序）：")
            print(_zscore_outlier_report(df, dtm0))
    elif not report.ok():
        print(f"[data_quality] ❌ {report.source} 检查失败: {report.errors}")
    if raise_on_error and not report.ok():
        raise ValueError(f"数据质量检查失败（{report.source}）：{report.errors}")


# ─────────────────────────────────────────────
# 便捷包装：直接替代 pd.read_csv + 检查
# ─────────────────────────────────────────────
def read_csv_with_check(
    filepath: str,
    dtm0: Optional[str] = None,
    required_cols: Optional[list[str]] = None,
    vol_cols: Optional[list[str]] = None,
    price_cols: Optional[list[str]] = None,
    nan_sentinel: float = -9999900,
    raise_on_error: bool = False,
    verbose: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    读取 CSV 并立即执行质量检查，与原代码替换方式保持一致：
        df = pd.read_csv(path).replace(-9999900, float('nan'))
    替换为：
        df = read_csv_with_check(path, dtm0='dtm_0', ...)
    """
    df = pd.read_csv(filepath, **read_csv_kwargs).replace(nan_sentinel, float("nan"))
    df = df[[c for c in df.columns if not c.startswith("unnamed")]]
    check_data_quality(
        df,
        source=str(filepath),
        dtm0=dtm0,
        required_cols=required_cols,
        vol_cols=vol_cols,
        price_cols=price_cols,
        nan_sentinel=nan_sentinel,  # 替换后应为 0，此处传入用于二次保险
        raise_on_error=raise_on_error,
        verbose=verbose,
    )
    return df
