"""Excel 操作工具集"""

from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

from .excel_loader import get_loader
from .config import get_config


def _limit_result(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    """限制返回结果行数"""
    config = get_config()
    if limit is None:
        limit = config.excel.default_result_limit
    limit = min(limit, config.excel.max_result_limit)
    return df.head(limit)


def _df_to_result(df: pd.DataFrame, limit: Optional[int] = None, select_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """将 DataFrame 转换为结果字典"""
    if select_columns:
        # 确保请求的列存在
        available_cols = [c for c in select_columns if c in df.columns]
        if available_cols:
            df = df[available_cols]
    
    limited_df = _limit_result(df, limit)
    return {
        "total_rows": len(df),
        "returned_rows": len(limited_df),
        "columns": list(limited_df.columns),
        "data": limited_df.to_dict(orient="records"),
    }


def _get_filter_mask(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.Series:
    """内部辅助函数：生成单个筛选条件的布尔掩码"""
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在，可用列: {list(df.columns)}")
    
    col = df[column]
    
    # 尝试将 value 转换为数值进行比较
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        numeric_value = None
    
    compare_value = numeric_value if numeric_value is not None else value
    
    if operator == "==":
        return col == compare_value
    elif operator == "!=":
        return col != compare_value
    elif operator == ">":
        return col > compare_value
    elif operator == "<":
        return col < compare_value
    elif operator == ">=":
        return col >= compare_value
    elif operator == "<=":
        return col <= compare_value
    elif operator == "contains":
        return col.astype(str).str.contains(str(value), case=False, na=False)
    elif operator == "startswith":
        return col.astype(str).str.startswith(str(value), na=False)
    elif operator == "endswith":
        return col.astype(str).str.endswith(str(value), na=False)
    else:
        raise ValueError(f"不支持的运算符: {operator}")


@tool
def filter_data(
    column: Optional[str] = None, 
    operator: Optional[str] = None, 
    value: Optional[Any] = None, 
    filters: Optional[List[Dict[str, Any]]] = None,
    select_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    limit: int = 20
) -> Dict[str, Any]:
    """按条件筛选 Excel 数据，支持排序。
    
    Args:
        column: 单条件筛选时的列名
        operator: 单条件筛选时的比较运算符
        value: 单条件筛选时的比较值（支持字符串、数值等任意类型）
        filters: 多条件筛选列表，每个元素为 {"column": "...", "operator": "...", "value": ...}
        select_columns: 指定返回的列名列表，为空则返回所有列
        sort_by: 排序列名，可选
        ascending: 排序方向，True为升序，False为降序，默认True
        limit: 返回结果数量限制，默认20
        
    Returns:
        筛选后的数据（可选排序）
    """
    loader = get_loader()
    df = loader.dataframe.copy()
    
    try:
        # 初始掩码为全 True
        final_mask = pd.Series([True] * len(df))
        
        # 1. 处理单条件参数 (兼容旧调用)
        if column and operator and value is not None:
            mask = _get_filter_mask(df, column, operator, value)
            final_mask &= mask
            
        # 2. 处理多条件列表
        if filters:
            for f in filters:
                f_col = f.get("column")
                f_op = f.get("operator")
                f_val = f.get("value")
                if f_col and f_op and f_val is not None:
                    mask = _get_filter_mask(df, f_col, f_op, f_val)
                    final_mask &= mask
        
        result_df = df[final_mask]
        
        # 3. 排序（如果指定了 sort_by）
        if sort_by:
            if sort_by not in result_df.columns:
                return {"error": f"排序列 '{sort_by}' 不存在，可用列: {list(result_df.columns)}"}
            result_df = result_df.sort_values(by=sort_by, ascending=ascending)
        
        return _df_to_result(result_df, limit, select_columns)
    except Exception as e:
        return {"error": f"筛选出错: {str(e)}"}


@tool
def aggregate_data(
    column: str, 
    agg_func: str,
    filters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """对指定列进行聚合统计。可选先筛选数据再聚合。
    
    Args:
        column: 要统计的列名
        agg_func: 聚合函数，可选值: sum, mean, count, min, max, median, std
        filters: 可选的筛选条件列表，每个元素为 {"column": "...", "operator": "...", "value": ...}
        
    Returns:
        统计结果
    """
    loader = get_loader()
    df = loader.dataframe.copy()
    
    # 如果有筛选条件，先进行筛选
    if filters:
        try:
            final_mask = pd.Series([True] * len(df))
            for f in filters:
                f_col = f.get("column")
                f_op = f.get("operator")
                f_val = f.get("value")
                if f_col and f_op and f_val is not None:
                    mask = _get_filter_mask(df, f_col, f_op, f_val)
                    final_mask &= mask
            df = df[final_mask]
        except Exception as e:
            return {"error": f"筛选条件错误: {str(e)}"}
    
    if column not in df.columns:
        return {"error": f"列 '{column}' 不存在，可用列: {list(df.columns)}"}
    
    col = df[column]
    
    try:
        if agg_func == "sum":
            result = col.sum()
        elif agg_func == "mean":
            result = col.mean()
        elif agg_func == "count":
            result = col.count()
        elif agg_func == "min":
            result = col.min()
        elif agg_func == "max":
            result = col.max()
        elif agg_func == "median":
            result = col.median()
        elif agg_func == "std":
            result = col.std()
        else:
            return {"error": f"不支持的聚合函数: {agg_func}"}
        
        # 处理 numpy 类型
        if hasattr(result, 'item'):
            result = result.item()
        
        return {
            "column": column,
            "function": agg_func,
            "filtered_rows": len(df),
            "result": result,
        }
    except Exception as e:
        return {"error": f"聚合计算出错: {str(e)}"}


@tool
def group_and_aggregate(
    group_by: str, 
    agg_column: str, 
    agg_func: str, 
    filters: Optional[List[Dict[str, Any]]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """按列分组并进行聚合统计。可选先筛选数据再分组。
    
    Args:
        group_by: 分组列名
        agg_column: 要聚合的列名
        agg_func: 聚合函数，可选值: sum, mean, count, min, max
        filters: 可选的筛选条件列表
        limit: 返回结果数量限制，默认20
        
    Returns:
        分组聚合结果
    """
    loader = get_loader()
    df = loader.dataframe.copy()
    
    # 如果有筛选条件，先进行筛选
    if filters:
        try:
            final_mask = pd.Series([True] * len(df))
            for f in filters:
                f_col = f.get("column")
                f_op = f.get("operator")
                f_val = f.get("value")
                if f_col and f_op and f_val is not None:
                    mask = _get_filter_mask(df, f_col, f_op, f_val)
                    final_mask &= mask
            df = df[final_mask]
        except Exception as e:
            return {"error": f"筛选条件错误: {str(e)}"}
    
    if group_by not in df.columns:
        return {"error": f"分组列 '{group_by}' 不存在，可用列: {list(df.columns)}"}
    if agg_column not in df.columns:
        return {"error": f"聚合列 '{agg_column}' 不存在，可用列: {list(df.columns)}"}
    
    try:
        grouped = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
        grouped.columns = [group_by, f"{agg_column}_{agg_func}"]
        
        # 按聚合结果降序排序
        grouped = grouped.sort_values(by=grouped.columns[1], ascending=False)
        
        result = _df_to_result(grouped, limit)
        result["filtered_rows"] = len(df)
        return result
    except Exception as e:
        return {"error": f"分组聚合出错: {str(e)}"}


@tool
def sort_data(
    column: str, 
    ascending: bool = True, 
    filters: Optional[List[Dict[str, Any]]] = None,
    select_columns: Optional[List[str]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """按指定列排序数据。可选先筛选、指定返回列。
    
    Args:
        column: 排序列名
        ascending: 是否升序排列，默认True
        filters: 可选的筛选条件列表
        select_columns: 指定返回的列名列表
        limit: 返回结果数量限制，默认20
        
    Returns:
        排序后的数据
    """
    loader = get_loader()
    df = loader.dataframe.copy()
    
    # 如果有筛选条件，先进行筛选
    if filters:
        try:
            final_mask = pd.Series([True] * len(df))
            for f in filters:
                f_col = f.get("column")
                f_op = f.get("operator")
                f_val = f.get("value")
                if f_col and f_op and f_val is not None:
                    mask = _get_filter_mask(df, f_col, f_op, f_val)
                    final_mask &= mask
            df = df[final_mask]
        except Exception as e:
            return {"error": f"筛选条件错误: {str(e)}"}
    
    if column not in df.columns:
        return {"error": f"列 '{column}' 不存在，可用列: {list(df.columns)}"}
    
    try:
        sorted_df = df.sort_values(by=column, ascending=ascending)
        return _df_to_result(sorted_df, limit, select_columns)
    except Exception as e:
        return {"error": f"排序出错: {str(e)}"}


@tool
def search_data(
    keyword: str, 
    columns: Optional[List[str]] = None,
    select_columns: Optional[List[str]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """在指定列或所有列中搜索关键词。
    
    Args:
        keyword: 搜索关键词
        columns: 要搜索的列名列表，为空则搜索所有列
        select_columns: 指定返回的列名列表
        limit: 返回结果数量限制，默认20
        
    Returns:
        包含关键词的数据行
    """
    loader = get_loader()
    df = loader.dataframe
    
    try:
        # 确定搜索范围
        search_cols = columns if columns else df.columns
        
        # 在指定列中搜索
        mask = pd.Series([False] * len(df))
        for col in search_cols:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
        
        result_df = df[mask]
        return _df_to_result(result_df, limit, select_columns)
    except Exception as e:
        return {"error": f"搜索出错: {str(e)}"}


@tool
def get_column_stats(
    column: str,
    filters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """获取指定列的详细统计信息。可选先筛选数据再统计。
    
    Args:
        column: 列名
        filters: 可选的筛选条件列表
        
    Returns:
        列的统计信息
    """
    loader = get_loader()
    df = loader.dataframe.copy()
    
    # 如果有筛选条件，先进行筛选
    if filters:
        try:
            final_mask = pd.Series([True] * len(df))
            for f in filters:
                f_col = f.get("column")
                f_op = f.get("operator")
                f_val = f.get("value")
                if f_col and f_op and f_val is not None:
                    mask = _get_filter_mask(df, f_col, f_op, f_val)
                    final_mask &= mask
            df = df[final_mask]
        except Exception as e:
            return {"error": f"筛选条件错误: {str(e)}"}
    
    if column not in df.columns:
        return {"error": f"列 '{column}' 不存在，可用列: {list(df.columns)}"}
    
    col = df[column]
    
    try:
        stats = {
            "column": column,
            "filtered_rows": len(df),
            "dtype": str(col.dtype),
            "count": int(col.count()),
            "null_count": int(col.isna().sum()),
            "unique_count": int(col.nunique()),
        }
        
        # 数值类型额外统计
        if pd.api.types.is_numeric_dtype(col):
            stats.update({
                "min": float(col.min()) if not col.isna().all() else None,
                "max": float(col.max()) if not col.isna().all() else None,
                "mean": float(col.mean()) if not col.isna().all() else None,
                "median": float(col.median()) if not col.isna().all() else None,
            })
        
        return stats
    except Exception as e:
        return {"error": f"统计出错: {str(e)}"}


@tool
def get_unique_values(
    column: str, 
    filters: Optional[List[Dict[str, Any]]] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """获取指定列的唯一值列表。可选先筛选数据。
    
    Args:
        column: 列名
        filters: 可选的筛选条件列表
        limit: 返回唯一值数量限制，默认50
        
    Returns:
        唯一值列表及其计数
    """
    loader = get_loader()
    df = loader.dataframe.copy()
    
    # 如果有筛选条件，先进行筛选
    if filters:
        try:
            final_mask = pd.Series([True] * len(df))
            for f in filters:
                f_col = f.get("column")
                f_op = f.get("operator")
                f_val = f.get("value")
                if f_col and f_op and f_val is not None:
                    mask = _get_filter_mask(df, f_col, f_op, f_val)
                    final_mask &= mask
            df = df[final_mask]
        except Exception as e:
            return {"error": f"筛选条件错误: {str(e)}"}
    
    if column not in df.columns:
        return {"error": f"列 '{column}' 不存在，可用列: {list(df.columns)}"}
    
    try:
        value_counts = df[column].value_counts()
        total_unique = len(value_counts)
        
        if limit:
            value_counts = value_counts.head(limit)
        
        values = [
            {"value": str(idx), "count": int(count)}
            for idx, count in value_counts.items()
        ]
        
        return {
            "column": column,
            "filtered_rows": len(df),
            "total_unique": total_unique,
            "returned_unique": len(values),
            "values": values,
        }
    except Exception as e:
        return {"error": f"获取唯一值出错: {str(e)}"}


@tool
def get_data_preview(n_rows: int = 10) -> Dict[str, Any]:
    """获取数据预览。
    
    Args:
        n_rows: 预览行数，默认10行
        
    Returns:
        数据预览
    """
    loader = get_loader()
    return loader.get_preview(n_rows)


@tool
def get_current_time() -> Dict[str, Any]:
    """获取当前系统时间。
    
    Returns:
        当前时间信息
    """
    from datetime import datetime
    now = datetime.now()
    return {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timestamp": now.timestamp()
    }


@tool
def calculate(expressions: List[str]) -> Dict[str, Any]:
    """执行数学计算。
    
    Args:
        expressions: 数学表达式列表，例如 ["(100+200)*0.5", "500/2"]
        
    Returns:
        每个表达式的计算结果
    """
    import math
    
    results = {}
    
    # 定义安全的计算环境
    safe_env = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "math": math,
    }
    
    for expr in expressions:
        try:
            # 移除危险字符，防止恶意代码
            if any(char in expr for char in ["__", "import", "eval", "exec", "open"]):
                results[expr] = "Error: Unsafe expression"
                continue
                
            # 执行计算
            result = eval(expr, {"__builtins__": None}, safe_env)
            results[expr] = result
        except Exception as e:
            results[expr] = f"Error: {str(e)}"
            
    return {"results": results}


# 导出工具列表
ALL_TOOLS = [
    filter_data,
    aggregate_data,
    group_and_aggregate,
    # sort_data,  # 已合并到 filter_data
    search_data,
    get_column_stats,
    get_unique_values,
    get_data_preview,
    get_current_time,
    calculate,
]
