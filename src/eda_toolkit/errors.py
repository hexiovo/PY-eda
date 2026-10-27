"""统一错误边界；数值函数不会把错误转换成零值或虚假的成功。"""

def fail(code, message, *, category="input", stage="", record_id="", field="", remediation=""):
    """抛出带普通字典诊断的异常，不需要业务类或继承层次。"""
    error = ValueError(message)
    error.eda_details = {"category": category, "code": code, "stage": stage,
                         "record_id": record_id, "field": field, "message": message,
                         "remediation": remediation}
    raise error


def error_to_dict(error, *, stage="", record_id=""):
    """保留可展示的错误字段；不向结果文件暴露完整本机堆栈。"""
    details = getattr(error, "eda_details", None)
    if details is not None:
        result = dict(details)
        result["stage"] = result["stage"] or stage
        result["record_id"] = result["record_id"] or record_id
        return result
    return {"category": "runtime", "code": type(error).__name__, "stage": stage,
            "record_id": record_id, "field": "", "message": str(error),
            "remediation": "检查输入与运行日志；修正后使用新的运行目录重试。"}
