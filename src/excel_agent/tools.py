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


@tool
def generate_chart(
    chart_type: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    group_by: Optional[str] = None,
    agg_func: str = "sum",
    title: str = "",
    filters: Optional[List[Dict[str, Any]]] = None,
    series_columns: Optional[List[str]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """生成 ECharts 可视化图表配置。
    
    Args:
        chart_type: 图表类型，可选: bar(柱状图), line(折线图), pie(饼图), 
                   scatter(散点图), radar(雷达图), funnel(漏斗图)。
                   为空或"auto"时自动推荐。
        x_column: X轴数据列名（分类轴）
        y_column: Y轴数据列名（数值轴，单系列时使用）
        group_by: 分组列名（用于饼图和多系列图）
        agg_func: 聚合函数: sum, mean, count, min, max
        title: 图表标题
        filters: 筛选条件列表
        series_columns: 多系列Y轴列名列表
        limit: 数据点数量限制，默认20
        
    Returns:
        包含 ECharts 配置的字典 {"chart": {...}, "message": "..."}
    """
    loader = get_loader()
    df = loader.dataframe.copy()
    
    # 应用筛选条件
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
    
    if len(df) == 0:
        return {"error": "筛选后无数据，无法生成图表"}
    
    # 自动推荐图表类型
    def recommend_chart_type() -> str:
        """根据数据特征推荐图表类型"""
        if group_by and y_column:
            # 分组场景：检查分组数量
            unique_groups = df[group_by].nunique() if group_by in df.columns else 0
            if unique_groups <= 8:
                return "pie"  # 少量分组适合饼图
            return "bar"  # 多分组适合柱状图
        
        if x_column and y_column:
            x_dtype = df[x_column].dtype if x_column in df.columns else None
            y_dtype = df[y_column].dtype if y_column in df.columns else None
            
            # 两个数值列 → 散点图
            if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
                return "scatter"
            
            # X轴是日期/时间类型 → 折线图
            if pd.api.types.is_datetime64_any_dtype(x_dtype):
                return "line"
            
            # 默认柱状图
            return "bar"
        
        # 仅有分组列 → 饼图
        if group_by:
            return "pie"
        
        return "bar"
    
    # 确定图表类型
    final_chart_type = chart_type if chart_type and chart_type != "auto" else recommend_chart_type()
    
    try:
        # 准备图表数据
        chart_data = _prepare_chart_data(df, final_chart_type, x_column, y_column, 
                                         group_by, agg_func, series_columns, limit)
        
        if "error" in chart_data:
            return chart_data
        
        # 生成 ECharts 配置
        chart_config = _build_echart_config(final_chart_type, chart_data, title)
        
        chart_type_names = {
            "bar": "柱状图", "line": "折线图", "pie": "饼图",
            "scatter": "散点图", "radar": "雷达图", "funnel": "漏斗图"
        }
        message = f"已生成{chart_type_names.get(final_chart_type, final_chart_type)}，共 {chart_data.get('data_count', 0)} 个数据点。"
        
        return {
            "chart": chart_config,
            "chart_type": final_chart_type,
            "message": message
        }
    except Exception as e:
        return {"error": f"生成图表出错: {str(e)}"}


def _prepare_chart_data(df: pd.DataFrame, chart_type: str, x_column: Optional[str],
                        y_column: Optional[str], group_by: Optional[str],
                        agg_func: str, series_columns: Optional[List[str]], 
                        limit: int) -> Dict[str, Any]:
    """准备图表数据"""
    
    if chart_type == "pie":
        # 饼图：按分组列聚合
        if group_by and group_by in df.columns:
            if y_column and y_column in df.columns:
                grouped = df.groupby(group_by)[y_column].agg(agg_func).reset_index()
                grouped.columns = ["name", "value"]
            else:
                grouped = df[group_by].value_counts().reset_index()
                grouped.columns = ["name", "value"]
            
            grouped = grouped.head(limit)
            data = [{"name": str(row["name"]), "value": float(row["value"])} 
                    for _, row in grouped.iterrows()]
            return {"data": data, "data_count": len(data)}
        else:
            return {"error": "饼图需要指定 group_by 分组列"}
    
    elif chart_type == "scatter":
        # 散点图：需要两个数值列
        if not x_column or not y_column:
            return {"error": "散点图需要指定 x_column 和 y_column"}
        if x_column not in df.columns or y_column not in df.columns:
            return {"error": f"列不存在: {x_column} 或 {y_column}"}
        
        scatter_df = df[[x_column, y_column]].dropna().head(limit * 5)  # 散点图可以多一些点
        data = scatter_df.values.tolist()
        return {
            "data": data, 
            "x_name": x_column, 
            "y_name": y_column,
            "data_count": len(data)
        }
    
    elif chart_type == "radar":
        # 雷达图：多个指标对比
        if not series_columns or len(series_columns) < 3:
            return {"error": "雷达图需要至少3个 series_columns 指标列"}
        
        valid_cols = [c for c in series_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(valid_cols) < 3:
            return {"error": "雷达图需要至少3个有效的数值列"}
        
        # 计算每个指标的聚合值
        if group_by and group_by in df.columns:
            # 按分组生成多个雷达系列
            grouped = df.groupby(group_by)[valid_cols].agg(agg_func).head(limit)
            indicators = [{"name": col, "max": float(df[col].max() * 1.2)} for col in valid_cols]
            series_data = []
            for name, row in grouped.iterrows():
                series_data.append({
                    "name": str(name),
                    "value": [float(row[col]) for col in valid_cols]
                })
            return {"indicators": indicators, "series": series_data, "data_count": len(series_data)}
        else:
            # 单系列雷达图
            indicators = [{"name": col, "max": float(df[col].max() * 1.2)} for col in valid_cols]
            values = [float(df[col].agg(agg_func)) for col in valid_cols]
            return {"indicators": indicators, "series": [{"name": "数据", "value": values}], "data_count": 1}
    
    elif chart_type == "funnel":
        # 漏斗图：类似饼图，按值降序排列
        if group_by and group_by in df.columns:
            if y_column and y_column in df.columns:
                grouped = df.groupby(group_by)[y_column].agg(agg_func).reset_index()
                grouped.columns = ["name", "value"]
            else:
                grouped = df[group_by].value_counts().reset_index()
                grouped.columns = ["name", "value"]
            
            grouped = grouped.sort_values("value", ascending=False).head(limit)
            data = [{"name": str(row["name"]), "value": float(row["value"])} 
                    for _, row in grouped.iterrows()]
            return {"data": data, "data_count": len(data)}
        else:
            return {"error": "漏斗图需要指定 group_by 分组列"}
    
    else:
        # bar / line：分类 + 数值
        if not x_column:
            return {"error": f"{chart_type}图需要指定 x_column"}
        if x_column not in df.columns:
            return {"error": f"列 '{x_column}' 不存在"}
        
        # 多系列处理
        if series_columns:
            valid_series = [c for c in series_columns if c in df.columns]
            if not valid_series:
                return {"error": "series_columns 中没有有效的列"}
            
            # 按 x_column 分组，计算每个系列的聚合值
            grouped = df.groupby(x_column)[valid_series].agg(agg_func).head(limit)
            categories = [str(idx) for idx in grouped.index]
            series = [
                {"name": col, "data": grouped[col].tolist()}
                for col in valid_series
            ]
            return {"categories": categories, "series": series, "data_count": len(categories)}
        
        # 单系列处理
        if y_column and y_column in df.columns:
            grouped = df.groupby(x_column)[y_column].agg(agg_func).reset_index()
            grouped.columns = ["category", "value"]
            grouped = grouped.sort_values("value", ascending=False).head(limit)
            categories = [str(c) for c in grouped["category"]]
            values = grouped["value"].tolist()
        else:
            # 仅计数
            grouped = df[x_column].value_counts().head(limit)
            categories = [str(idx) for idx in grouped.index]
            values = grouped.values.tolist()
        
        return {"categories": categories, "values": values, "data_count": len(categories)}


def _build_echart_config(chart_type: str, data: Dict[str, Any], title: str) -> Dict[str, Any]:
    """构建 ECharts 配置"""
    
    # 通用配置
    base_config = {
        "title": {
            "text": title,
            "left": "center",
            "textStyle": {"color": "#e5e7eb"}
        },
        "tooltip": {"trigger": "item" if chart_type in ["pie", "scatter", "funnel"] else "axis"},
        "backgroundColor": "transparent"
    }
    
    if chart_type == "pie":
        return {
            **base_config,
            "legend": {
                "orient": "vertical",
                "left": "left",
                "textStyle": {"color": "#9ca3af"}
            },
            "series": [{
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": True,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#1f2937",
                    "borderWidth": 2
                },
                "label": {"color": "#e5e7eb"},
                "emphasis": {
                    "label": {"show": True, "fontSize": 16, "fontWeight": "bold"}
                },
                "data": data["data"]
            }]
        }
    
    elif chart_type == "scatter":
        return {
            **base_config,
            "xAxis": {
                "type": "value",
                "name": data.get("x_name", ""),
                "axisLabel": {"color": "#9ca3af"},
                "axisLine": {"lineStyle": {"color": "#4b5563"}}
            },
            "yAxis": {
                "type": "value",
                "name": data.get("y_name", ""),
                "axisLabel": {"color": "#9ca3af"},
                "axisLine": {"lineStyle": {"color": "#4b5563"}}
            },
            "series": [{
                "type": "scatter",
                "symbolSize": 10,
                "data": data["data"],
                "itemStyle": {"color": "#6366f1"}
            }]
        }
    
    elif chart_type == "radar":
        return {
            **base_config,
            "legend": {
                "data": [s["name"] for s in data["series"]],
                "bottom": 0,
                "textStyle": {"color": "#9ca3af"}
            },
            "radar": {
                "indicator": data["indicators"],
                "axisName": {"color": "#9ca3af"},
                "splitLine": {"lineStyle": {"color": "#4b5563"}},
                "splitArea": {"areaStyle": {"color": ["rgba(99,102,241,0.1)", "rgba(99,102,241,0.05)"]}}
            },
            "series": [{
                "type": "radar",
                "data": data["series"]
            }]
        }
    
    elif chart_type == "funnel":
        return {
            **base_config,
            "legend": {
                "data": [d["name"] for d in data["data"]],
                "bottom": 0,
                "textStyle": {"color": "#9ca3af"}
            },
            "series": [{
                "type": "funnel",
                "left": "10%",
                "width": "80%",
                "label": {"show": True, "position": "inside", "color": "#fff"},
                "labelLine": {"show": False},
                "itemStyle": {"borderColor": "#1f2937", "borderWidth": 1},
                "emphasis": {"label": {"fontSize": 16}},
                "data": data["data"]
            }]
        }
    
    else:
        # bar / line
        config = {
            **base_config,
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": data["categories"],
                "axisLabel": {"color": "#9ca3af", "rotate": 30 if len(data["categories"]) > 8 else 0},
                "axisLine": {"lineStyle": {"color": "#4b5563"}}
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"color": "#9ca3af"},
                "axisLine": {"lineStyle": {"color": "#4b5563"}},
                "splitLine": {"lineStyle": {"color": "#374151"}}
            }
        }
        
        # 处理多系列
        if "series" in data:
            config["legend"] = {
                "data": [s["name"] for s in data["series"]],
                "bottom": 0,
                "textStyle": {"color": "#9ca3af"}
            }
            config["series"] = [
                {
                    "name": s["name"],
                    "type": chart_type,
                    "data": s["data"],
                    "smooth": chart_type == "line"
                }
                for s in data["series"]
            ]
        else:
            config["series"] = [{
                "type": chart_type,
                "data": data["values"],
                "smooth": chart_type == "line",
                "itemStyle": {"color": "#6366f1"},
                "areaStyle": {"color": "rgba(99,102,241,0.2)"} if chart_type == "line" else None
            }]
        
        return config


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
    generate_chart,
]
