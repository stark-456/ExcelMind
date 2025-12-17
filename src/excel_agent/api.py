"""FastAPI HTTP 服务（支持流式输出和多表管理）"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from .config import get_config, load_config, set_config
from .excel_loader import get_loader, reset_loader
from .graph import get_graph, reset_graph
from .stream import stream_chat

import tempfile
import os

# 创建 FastAPI 应用
app = FastAPI(
    title="Excel 智能问数 Agent",
    description="基于 LangGraph 的 Excel 数据分析助手 API（支持多表）",
    version="0.2.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico")
async def favicon():
    """返回空 favicon 避免 404"""
    return Response(status_code=204)


class LoadExcelRequest(BaseModel):
    """加载 Excel 请求"""
    file_path: str
    sheet_name: Optional[str] = None


class LoadExcelResponse(BaseModel):
    """加载 Excel 响应"""
    success: bool
    message: str
    table_id: Optional[str] = None
    structure: Optional[Dict[str, Any]] = None
    preview: Optional[Dict[str, Any]] = None
    tables: Optional[List[Dict[str, Any]]] = None  # 所有表列表


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    history: Optional[list] = None  # 历史对话列表


class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool
    response: str
    tool_calls: Optional[list] = None


class TableInfo(BaseModel):
    """表信息"""
    id: str
    filename: str
    sheet_name: str
    total_rows: int
    total_columns: int
    loaded_at: str
    is_active: bool


class StatusResponse(BaseModel):
    """状态响应"""
    excel_loaded: bool
    tables: Optional[List[Dict[str, Any]]] = None  # 所有表信息
    active_table_id: Optional[str] = None
    active_table: Optional[Dict[str, Any]] = None  # 当前活跃表详情


class SetActiveTableRequest(BaseModel):
    """设置活跃表请求"""
    table_id: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
    <html>
        <head><title>Excel Agent</title></head>
        <body>
            <h1>Excel 智能问数 Agent</h1>
            <p>API 文档: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """)


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取当前状态（包含多表信息）"""
    loader = get_loader()
    
    if not loader.is_loaded:
        return StatusResponse(excel_loaded=False)
    
    # 获取所有表信息
    tables = loader.list_tables()
    
    # 获取活跃表详情
    active_loader = loader.get_active_loader()
    active_table = None
    if active_loader:
        structure = active_loader.get_structure()
        active_table = {
            "file_path": structure["file_path"],
            "sheet_name": structure["sheet_name"],
            "total_rows": structure["total_rows"],
            "total_columns": structure["total_columns"],
            "columns": structure["columns"],
        }
    
    return StatusResponse(
        excel_loaded=True,
        tables=tables,
        active_table_id=loader.active_table_id,
        active_table=active_table,
    )


@app.get("/tables")
async def list_tables():
    """获取所有已加载的表列表"""
    loader = get_loader()
    return {
        "success": True,
        "tables": loader.list_tables(),
        "active_table_id": loader.active_table_id,
    }


@app.put("/tables/active")
async def set_active_table(request: SetActiveTableRequest):
    """设置当前活跃表"""
    loader = get_loader()
    
    if not loader.set_active_table(request.table_id):
        raise HTTPException(status_code=404, detail=f"表不存在: {request.table_id}")
    
    # 重置图以使用新的活跃表数据
    reset_graph()
    
    # 获取新活跃表的信息
    active_loader = loader.get_active_loader()
    structure = active_loader.get_structure() if active_loader else None
    preview = active_loader.get_preview() if active_loader else None
    
    return {
        "success": True,
        "message": f"已切换到表: {request.table_id}",
        "structure": structure,
        "preview": preview,
        "tables": loader.list_tables(),
    }


@app.delete("/tables/{table_id}")
async def delete_table(table_id: str):
    """删除指定表"""
    loader = get_loader()
    
    if not loader.remove_table(table_id):
        raise HTTPException(status_code=404, detail=f"表不存在: {table_id}")
    
    # 重置图
    reset_graph()
    
    return {
        "success": True,
        "message": f"已删除表: {table_id}",
        "tables": loader.list_tables(),
        "active_table_id": loader.active_table_id,
    }


@app.get("/tables/{table_id}/columns")
async def get_table_columns(table_id: str):
    """获取指定表的列名列表"""
    loader = get_loader()
    columns = loader.get_table_columns(table_id)
    
    if not columns:
        raise HTTPException(status_code=404, detail=f"表不存在或无数据: {table_id}")
    
    return {
        "success": True,
        "table_id": table_id,
        "columns": columns,
    }


class JoinTablesRequest(BaseModel):
    """连接表请求（支持多字段连接）"""
    table1_id: str
    table2_id: str
    keys1: List[str]  # 表1的连接字段列表
    keys2: List[str]  # 表2的连接字段列表
    join_type: str = "inner"
    new_name: str


@app.post("/tables/join")
async def join_tables(request: JoinTablesRequest):
    """连接两张表（支持多字段连接）"""
    loader = get_loader()
    
    try:
        table_id, structure = loader.join_tables(
            table1_id=request.table1_id,
            table2_id=request.table2_id,
            keys1=request.keys1,
            keys2=request.keys2,
            join_type=request.join_type,
            new_name=request.new_name,
        )
        
        # 重置图
        reset_graph()
        
        return {
            "success": True,
            "message": f"成功创建连接表: {request.new_name}",
            "table_id": table_id,
            "structure": structure,
            "tables": loader.list_tables(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"连接失败: {str(e)}")


class SuggestJoinRequest(BaseModel):
    """AI联表建议请求"""
    table1_id: str
    table2_id: str


@app.post("/tables/suggest-join")
async def suggest_join(request: SuggestJoinRequest):
    """使用AI分析两表结构并建议联表配置"""
    from .join_service import suggest_join_config
    
    loader = get_loader()
    
    # 获取两表的loader
    loader1 = loader.get_table(request.table1_id)
    loader2 = loader.get_table(request.table2_id)
    
    if not loader1 or not loader2:
        raise HTTPException(status_code=404, detail="指定的表不存在")
    
    # 获取两表的summary
    table1_summary = loader1.get_summary()
    table2_summary = loader2.get_summary()
    
    try:
        suggestion = suggest_join_config(table1_summary, table2_summary)
        return {
            "success": True,
            "suggestion": suggestion,
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        print(f"[AI建议] 异常: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI分析失败: {str(e)}")


@app.post("/load", response_model=LoadExcelResponse)
async def load_excel(request: LoadExcelRequest):
    """通过文件路径加载 Excel 文件（追加模式）"""
    try:
        loader = get_loader()
        table_id, structure = loader.add_table(request.file_path, request.sheet_name)
        
        # 获取预览
        active_loader = loader.get_active_loader()
        preview = active_loader.get_preview() if active_loader else None
        
        # 重置图以使用新的 Excel 数据
        reset_graph()
        
        return LoadExcelResponse(
            success=True,
            message=f"成功加载 Excel 文件: {request.file_path}",
            table_id=table_id,
            structure=structure,
            preview=preview,
            tables=loader.list_tables(),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载失败: {str(e)}")


@app.post("/upload", response_model=LoadExcelResponse)
async def upload_excel(file: UploadFile = File(...), sheet_name: Optional[str] = None):
    """上传 Excel 文件（追加模式，会添加到多表管理器）"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")
    
    # 检查文件扩展名
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ['.xlsx', '.xls', '.xlsm']:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")
    
    try:
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 加载 Excel（追加模式）
        loader = get_loader()
        table_id, structure = loader.add_table(tmp_path, sheet_name)
        
        # 更新文件名（临时文件路径替换为原始文件名）
        table_info = loader.get_table_info(table_id)
        if table_info:
            table_info.filename = file.filename
        
        # 获取预览
        active_loader = loader.get_active_loader()
        preview = active_loader.get_preview() if active_loader else None
        
        # 重置图
        reset_graph()
        
        return LoadExcelResponse(
            success=True,
            message=f"成功上传并加载 Excel 文件: {file.filename}",
            table_id=table_id,
            structure=structure,
            preview=preview,
            tables=loader.list_tables(),
        )
    except Exception as e:
        # 清理临时文件
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """与 Agent 对话（非流式）"""
    loader = get_loader()
    
    if not loader.is_loaded:
        raise HTTPException(
            status_code=400, 
            detail="请先加载 Excel 文件（使用 /load 或 /upload 接口）"
        )
    
    try:
        graph = get_graph()
        
        # 构建输入
        inputs = {
            "messages": [HumanMessage(content=request.message)],
            "is_relevant": True,
        }
        
        # 执行图
        result = graph.invoke(inputs)
        
        # 提取响应
        messages = result.get("messages", [])
        response_text = ""
        tool_calls = []
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    response_text = msg.content
                if msg.tool_calls:
                    tool_calls.extend([
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in msg.tool_calls
                    ])
        
        return ChatResponse(
            success=True,
            response=response_text,
            tool_calls=tool_calls if tool_calls else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """与 Agent 对话（流式输出）"""
    loader = get_loader()
    
    if not loader.is_loaded:
        raise HTTPException(
            status_code=400, 
            detail="请先加载 Excel 文件"
        )
    
    # 获取当前活跃表ID，用于前端标记消息
    active_table_id = loader.active_table_id
    
    async def generate():
        # 先发送表ID信息
        yield f"data: {json.dumps({'type': 'table_info', 'table_id': active_table_id}, ensure_ascii=False)}\n\n"
        
        async for event in stream_chat(request.message, request.history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/reset")
async def reset():
    """重置 Agent 状态（清空所有表）"""
    reset_loader()
    reset_graph()
    return {"success": True, "message": "已重置 Agent 状态，所有表已清空"}


def run_server():
    """运行服务器"""
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )
