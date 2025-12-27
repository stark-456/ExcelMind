"""流式对话 - 使用手动工具解析避免 LangChain 工具绑定兼容性问题"""

import json
import re
from typing import Any, AsyncGenerator, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import get_config
from .excel_loader import get_loader
from .knowledge_base import get_knowledge_base, format_knowledge_context

from .tools import ALL_TOOLS


class CustomJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 Pandas/Numpy 类型"""
    
    def default(self, obj):
        # 处理 Pandas Timestamp
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # 处理 numpy 类型
        if hasattr(obj, 'item'):
            return obj.item()
        # 处理 numpy 数组
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        # 处理 pandas NaT
        if str(obj) == 'NaT':
            return None
        # 处理 pandas NA
        if str(obj) == '<NA>':
            return None
        return super().default(obj)


def json_dumps(obj, **kwargs):
    """使用自定义编码器的 JSON 序列化函数"""
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)


# 构建工具描述
TOOLS_DESCRIPTION = """
## 可用工具

你可以使用以下工具来分析 Excel 数据。当需要使用工具时，请使用以下 JSON 格式：

```json
{"tool": "工具名", "args": {"参数名": "参数值"}}
```

### 工具列表：

1. **filter_data** - 按条件筛选数据 (支持排序、指定列)
   - filters (list): 多条件筛选列表，每项包含 column, operator, value
   - select_columns (list): 指定返回的列名列表(可选)
   - sort_by (string): 排序列名(可选)，可一步完成筛选+排序
   - ascending (bool): 排序方向，true升序/false降序，默认true
   - column (string): 单条件筛选列名(可选)
   - operator (string): 比较运算符 (==, !=, >, <, >=, <=, contains, startswith, endswith)
   - value (任意类型): 比较值，支持字符串、数值、日期等
   - limit (int): 返回数量限制，默认20
   - **提示**: 需要筛选+排序时，直接用此工具一步完成，无需调用两次

2. **aggregate_data** - 对列进行聚合统计 (支持筛选后聚合)
   - column (string): 【必填】要统计的列名
   - agg_func (string): 【必填】聚合函数，必须指定为: sum(求和), mean(平均), count(计数), min, max, median, std
   - filters (list): 可选的筛选条件列表，先筛选再聚合

3. **group_and_aggregate** - 按列分组并聚合统计 (支持筛选)
   - group_by (string): 分组列名
   - agg_column (string): 要聚合的列名
   - agg_func (string): 聚合函数 (sum, mean, count, min, max)
   - filters (list): 筛选条件列表。**【重要】如果用户指定了日期、地区等条件，必须在此传入，否则会统计全表数据**
   - limit (int): 返回数量限制，默认20

4. **search_data** - 在指定列或所有列中搜索关键词
   - keyword (string): 搜索关键词
   - columns (list): 限制搜索的列名列表(可选)
   - select_columns (list): 指定返回的列名列表

   - limit (int): 返回数量限制，默认20

5. **get_column_stats** - 获取列的详细统计信息 (支持筛选)
   - column (string): 列名
   - filters (list): 可选的筛选条件列表

6. **get_unique_values** - 获取列的唯一值列表 (支持筛选)
   - column (string): 列名
   - filters (list): 可选的筛选条件列表
   - limit (int): 返回数量限制，默认50

7. **get_data_preview** - 获取数据预览
   - n_rows (int): 预览行数，默认10

8. **get_current_time** - 获取当前系统时间
   - 无参数

9. **calculate** - 执行数学计算 (支持批量)
    - expressions (list): 字符串格式的数学表达式列表，例如 ["(A+B)/C", "100*0.5"]

10. **generate_chart** - 生成 ECharts 可视化图表
    - chart_type (string): 图表类型，可选: bar(柱状图), line(折线图), pie(饼图), scatter(散点图), radar(雷达图), funnel(漏斗图)，或 "auto" 自动推荐
    - x_column (string): X轴数据列名（柱状图/折线图必填）
    - y_column (string): Y轴数据列名（数值列）
    - group_by (string): 分组列名（饼图/漏斗图必填）
    - agg_func (string): 聚合函数: sum, mean, count, min, max
    - title (string): 图表标题
    - filters (list): 筛选条件列表
    - series_columns (list): 多系列Y轴列名列表（雷达图需要至少3个）
    - limit (int): 数据点数量限制，默认20
    - **使用场景**: 用户想要可视化数据、生成图表、绘制趋势图、展示占比等需求时使用


## 重要规则
- 如果需要调用工具，只输出一个 JSON 对象，不要有其他文字
- 工具调用后我会告诉你结果，然后你再根据结果回答用户问题
- 如果不需要工具，直接用自然语言回答
"""


SYSTEM_PROMPT_WITH_TOOLS = """你是一个专业的 Excel 数据分析助手。

## 当前 Excel 信息
{excel_summary}

## 相关知识参考
{knowledge_context}

{tools_description}

## 工作原则
1. 根据用户问题，判断是否需要使用工具
2. 如需工具，**只输出**工具调用 JSON，**严禁**包含任何其他文字、思考过程或解释
3. 工具调用成功后，根据结果回答用户问题
4. **最终回答直接给出结论和分析**，不要描述"我使用了xx工具"或"我进行了xx操作"等内部过程
5. 回答语气友好，使用中文，并给出自己的一些数据分析建议
6. 如果有相关知识参考，请遵循其中的规则和建议
"""





def get_llm():
    """获取 LLM 实例"""
    config = get_config()
    provider = config.model.get_active_provider()
    return ChatOpenAI(
        model=provider.model_name,
        api_key=provider.api_key,
        base_url=provider.base_url if provider.base_url else None,
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
    )


def parse_tool_call(text: str) -> Dict[str, Any] | None:
    """从文本中解析工具调用 JSON（支持嵌套结构）"""
    # 尝试匹配 JSON 代码块
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取完整的 JSON 对象（支持嵌套）
    # 找到第一个包含 "tool" 的 { 开始，然后匹配括号
    start_idx = text.find('{')
    while start_idx != -1:
        # 尝试从这个位置提取完整JSON
        depth = 0
        end_idx = start_idx
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        
        if depth == 0 and end_idx > start_idx:
            candidate = text[start_idx:end_idx]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # 继续找下一个 {
        start_idx = text.find('{', start_idx + 1)
    
    return None


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """执行工具调用"""
    for tool in ALL_TOOLS:
        if tool.name == tool_name:
            try:
                return tool.invoke(tool_args)
            except Exception as e:
                return {"error": str(e)}
    return {"error": f"未找到工具: {tool_name}"}


async def stream_chat(message: str, history: list = None) -> AsyncGenerator[Dict[str, Any], None]:
    """执行对话
    
    Args:
        message: 当前用户消息
        history: 历史对话列表，每项为 {"role": "user"|"assistant", "content": "..."}
    """
    loader = get_loader()
    
    if not loader.is_loaded:
        yield {"type": "error", "content": "请先上传 Excel 文件"}
        return
    
    try:
        excel_summary = loader.get_summary()
        llm = get_llm()
        
        # 主对话
        yield {"type": "thinking", "content": "正在规划解答..."}
        
        # 检索相关知识
        knowledge_context = "暂无相关知识参考。"
        kb = get_knowledge_base()
        if kb:
            try:
                stats = kb.get_stats()
                print(f"[知识库] 状态: {stats['total_entries']} 条知识")
                relevant_knowledge = kb.search(query=message)
                print(f"[知识库] 检索到 {len(relevant_knowledge)} 条相关知识")
                if relevant_knowledge:
                    knowledge_context = format_knowledge_context(relevant_knowledge)
                    yield {"type": "thinking", "content": f"找到 {len(relevant_knowledge)} 条相关知识参考..."}
            except Exception as e:
                # 知识库检索失败不影响主流程
                print(f"[知识库检索] 警告: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[知识库] 未启用或初始化失败")
        
        system_prompt = SYSTEM_PROMPT_WITH_TOOLS.format(
            excel_summary=excel_summary,
            tools_description=TOOLS_DESCRIPTION,
            knowledge_context=knowledge_context
        )
        
        # 构建对话上下文，包含历史记录
        conversation = [SystemMessage(content=system_prompt)]
        
        # 获取当前活跃表信息
        active_table_info = loader.get_active_table_info()
        current_table_name = active_table_info.filename if active_table_info else "未知表"
        
        # 添加历史对话（包含表名标记）- 临时禁用，每次对话只关注本次
        if False:  # 原为 if history:
            from langchain_core.messages import AIMessage
            for h in history:
                content = h.get("content", "")
                table_name = h.get("tableName", "")
                
                # 如果历史消息有表名，且与当前表不同，添加标记
                if table_name and h.get("role") == "user":
                    content = f"[针对表: {table_name}] {content}"
                
                if h.get("role") == "user":
                    conversation.append(HumanMessage(content=content))
                elif h.get("role") == "assistant":
                    conversation.append(AIMessage(content=content))
        
        # 添加当前消息（标记当前表）
        current_message = f"[当前操作表: {current_table_name}] {message}"
        conversation.append(HumanMessage(content=current_message))
        
        # 更新 prompt 允许简短分析
        conversation[0].content += """
请严格遵循以下步骤：
1. **思考分析**：先进行简短的数据分析思路整理（Chain of Thought），解释为什么要使用该工具。这是一步非常关键的步骤。
2. **工具调用**：换行输出工具调用 JSON。
3. **最终回答**：在根据工具结果回答时，**直接给出结论**，不要复述第1步的思考过程，也不要提及使用了什么工具。
"""
        
        max_iterations = 50
        
        for _ in range(max_iterations):
            response = await llm.ainvoke(conversation)
            response_text = response.content
            
            # 解析工具调用
            tool_call = parse_tool_call(response_text)
            
            if tool_call and "tool" in tool_call:
                # 尝试提取 JSON 之前的思考文本
                thought_text = ""
                json_start = response_text.find('{')
                if json_match := re.search(r'```json', response_text):
                    thought_text = response_text[:json_match.start()].strip()
                elif json_start > 0:
                    thought_text = response_text[:json_start].strip()
                
                # 如果有思考文本且长度足够，发送更新
                if thought_text and len(thought_text) > 2:
                    yield {"type": "thinking", "content": thought_text}
                
                yield {"type": "thinking_done"}

                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})
                
                yield {
                    "type": "tool_call",
                    "name": tool_name,
                    "args": tool_args,
                }
                
                # 执行工具
                tool_result = execute_tool(tool_name, tool_args)
                
                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "result": tool_result,
                }
                
                # 将工具结果作为新消息继续对话
                result_message = f"工具 {tool_name} 执行结果：\n```json\n{json_dumps(tool_result, ensure_ascii=False, indent=2)}\n```\n\n请根据这个结果回答用户的问题。"
                
                conversation.append(response)
                conversation.append(HumanMessage(content=result_message))
                
            else:
                # 没有工具调用，直接输出响应
                yield {"type": "token", "content": response_text}
                yield {"type": "done", "content": response_text}
                return
        
        yield {"type": "error", "content": "达到最大迭代次数"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "thinking_done"}
        yield {"type": "error", "content": f"处理出错: {str(e)}"}
