---
id: kb_date_format_001
title: 账期字段格式说明
category: field_rule
tags:
  - 账期
  - 日期格式
  - ACCT_MONTH
  - 统计月份
related_columns:
  - 账期
  - ACCT_MONTH
  - 统计月份
  - 月份
priority: high
---

# 账期字段格式说明

## 规则描述

账期字段通常以 `YYYYMM` 或 `YYYYMMDD` 格式存储，例如：
- `202511` 表示 2025 年 11 月
- `20251104` 表示 2025 年 11 月 4 日

## 注意事项

1. **使用前缀匹配**：当用户查询某月数据时（如"2025年11月"），应使用 `startswith` 操作符匹配 `202511`，而不是精确匹配
2. **避免精确匹配**：不要使用 `==` 操作符匹配完整日期，因为数据可能包含日期后缀
3. **模糊匹配兼容**：如果用户输入的日期格式不完整，应尝试使用 `contains` 或 `startswith` 进行模糊匹配

## 常见变体

| 格式 | 示例 | 说明 |
|------|------|------|
| `YYYYMM` | `202511` | 标准年月格式 |
| `YYYYMMDD` | `20251104` | 完整日期格式 |
| `YYYY-MM` | `2025-11` | 带分隔符格式 |
| `YYYY/MM` | `2025/11` | 斜杠分隔格式 |

## 正确示例

当用户问"2025年11月的数据"时，应生成：
```json
{"tool": "filter_data", "args": {"column": "账期", "operator": "startswith", "value": "202511"}}
```

而不是：
```json
{"tool": "filter_data", "args": {"column": "账期", "operator": "==", "value": "202511"}}
```
