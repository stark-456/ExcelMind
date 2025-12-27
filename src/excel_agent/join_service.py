"""AI智能联表服务"""

import json
import re
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI

from .config import get_config
from .prompts import JOIN_SUGGEST_PROMPT


def get_llm():
    """获取 LLM 实例（与graph.py一致）"""
    config = get_config()
    provider = config.model.get_active_provider()
    return ChatOpenAI(
        model=provider.model_name,
        api_key=provider.api_key,
        base_url=provider.base_url if provider.base_url else None,
        temperature=0.1,  # 低温度以获得更稳定的JSON输出
    )


def suggest_join_config(table1_summary: str, table2_summary: str) -> Dict[str, Any]:
    """
    使用AI分析两表结构并建议联表配置
    
    Args:
        table1_summary: 表1的摘要信息
        table2_summary: 表2的摘要信息
        
    Returns:
        建议配置字典 {new_name, keys1, keys2, join_type, reason}
        
    Raises:
        ValueError: AI返回格式错误或缺少必要字段
    """
    # 构建prompt
    prompt = JOIN_SUGGEST_PROMPT.format(
        table1_summary=table1_summary,
        table2_summary=table2_summary
    )
    
    # 调用LLM
    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content
    
    print(f"[AI建议] LLM返回内容: {content[:500]}...")  # 调试日志
    
    # 解析JSON（从markdown代码块中提取）
    json_str = None
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接解析整个内容中的JSON对象
        json_match2 = re.search(r'\{[\s\S]*\}', content)
        if json_match2:
            json_str = json_match2.group(0)
        else:
            json_str = content
    
    try:
        suggestion = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[AI建议] JSON解析失败: {e}, 内容: {json_str[:200] if json_str else 'N/A'}")
        raise ValueError(f"AI返回格式解析失败: {str(e)}")
    
    # 验证必要字段
    required_fields = ['new_name', 'keys1', 'keys2', 'join_type']
    for field in required_fields:
        if field not in suggestion:
            raise ValueError(f"AI返回缺少必要字段: {field}")
    
    return suggestion
