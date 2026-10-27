"""稳定的动态数值指标标识符格式。"""

from decimal import Decimal


def autocorrelation_feature_id(lag_s):
    """返回计划约定的秒级滞后 ID（整数 `1s`、小数 `0_5s`）。"""
    token = format(Decimal(str(float(lag_s))).normalize(), "f")
    if "." in token:
        token = token.rstrip("0").rstrip(".").replace(".", "_")
    return f"clean_autocorr_lag_{token}s"
