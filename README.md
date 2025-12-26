# ExcelMind 📊

基于 LangGraph 的 **Excel 数据智能分析助手**，支持自然语言查询、多轮对话、流式输出、**ECharts 图表可视化**和可视化思考过程。

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange)

## ✨ 功能亮点

### 🎯 核心能力
- **自然语言查询**: 用中文直接提问，无需编写代码或公式
- **多轮对话**: 支持上下文关联的连续追问（如"和上个月相比呢？"）
- **流式输出**: 实时显示 AI 思考过程和回答，响应更流畅
- **智能工具调用**: 自动选择合适的数据分析工具，展示完整推理链路

### 🛠️ 丰富的数据分析工具
| 工具 | 功能 | 特性 |
|------|------|------|
| `filter_data` | 筛选+排序 | 支持多条件 AND、排序、指定返回列 |
| `aggregate_data` | 聚合统计 | 支持先筛选再聚合 |
| `group_and_aggregate` | 分组聚合 | 支持筛选后分组 |
| `search_data` | 关键词搜索 | 可限制搜索范围 |
| `get_column_stats` | 列统计 | 支持筛选后统计 |
| `get_unique_values` | 唯一值 | 支持筛选后获取 |
| `get_data_preview` | 数据预览 | 快速查看数据 |
| `get_current_time` | 获取时间 | 处理相对时间查询 |
| `calculate` | 数学计算 | 批量精确计算 |
| `generate_chart` | **图表生成** | ECharts 可视化，AI 自动推荐图表类型 |

### 🔄 多表协同
- **多表管理**: 同时上传和管理多个 Excel 表格
- **智能联表**: AI 自动分析表结构，通过 `🤖 智能联表` 功能一键生成连接建议
- **灵活连接**: 支持多字段（复合键）连接，以及 Inner/Left/Right/Outer 等多种连接方式
- **上下文感知**: 对话时明确显示当前所在的表格上下文

### 📚 本地知识库
- **私有知识存储**: 存储业务规则、字段说明、操作指南等私有知识
- **向量检索**: 基于 Chroma 向量数据库，使用 Embedding 模型进行语义检索
- **智能召回**: 对话时自动检索相关知识，注入到 Prompt 提升回答质量
- **Web 管理**: 右侧面板可视化管理知识条目，支持增删改查和文件上传
- **持久化存储**: 知识向量化后自动持久化，重启不丢失

### 📈 ECharts 图表可视化 (NEW)
- **多图表类型**: 支持柱状图、折线图、饼图、散点图、雷达图、漏斗图
- **AI 自动推荐**: 根据数据特征智能推荐最合适的图表类型
- **交互式图表**: 基于 ECharts 5.5，支持悬停提示、图例切换、响应式布局
- **自然语言触发**: 直接说"帮我画个图表"或"可视化销售数据"即可生成

### 🦺 现代化 Web 界面
- **侧边栏管理**: 清晰的表格列表和操作入口
- **拖拽上传**: 支持多文件拖拽上传，带进度提示
- **实时预览**: 上传后即时显示数据结构和预览
- **思考可视化**: 展示 AI 的推理过程（Chain of Thought）
- **工具调用展示**: 透明显示每一步工具调用和结果（美化版）
- **Markdown 渲染**: 完美支持表格、代码块等格式

### 🛡️ 安全与稳定
- **意图过滤**: 自动拒绝与 Excel 数据无关的闲聊
- **类型兼容**: 工具参数支持多种数据类型（字符串、数值、日期）
- **模糊匹配**: 日期字段支持前缀匹配（如 "202511" 匹配 "20251104"）
- **高迭代限制**: 支持复杂任务的多步推理（最多 50 次工具调用）

## 🚀 快速开始

### 环境要求
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (推荐) 或 pip

### 1. 克隆项目

```bash
git clone https://github.com/stark-456/ExcelMind.git
cd ExcelMind
```

### 2. 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -e .
```

### 3. 配置

编辑 `config.yaml` 配置模型参数：

```yaml
model:
  provider: "openai"
  model_name: "gpt-4"  # 或其他兼容模型
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"  # 可选，自定义端点
  temperature: 0.1
  max_tokens: 4096

excel:
  max_preview_rows: 20
  default_result_limit: 20
  max_result_limit: 1000

server:
  host: "0.0.0.0"
  port: 8000
```

也可使用环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-base-url"
```

### 4. 启动服务

```bash
# Web 服务模式（推荐）
uv run python -m excel_agent.main serve

# 命令行模式
uv run python -m excel_agent.main cli --excel your_file.xlsx
```

### 5. 使用

打开浏览器访问 `http://localhost:8000`：
1. 拖拽或点击上传 Excel 文件
2. 在聊天框输入自然语言问题
3. 查看 AI 的思考过程和分析结果

## 📡 API 接口

启动服务后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

### 主要接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/upload` | POST | 上传 Excel 文件 |
| `/load` | POST | 通过路径加载 Excel |
| `/chat/stream` | POST | 流式对话（推荐） |
| `/chat` | POST | 非流式对话 |
| `/status` | GET | 获取当前状态 |
| `/reset` | POST | 重置 Agent |

### 请求示例

```bash
# 上传 Excel
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_file.xlsx"

# 流式对话（带历史）
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "2025年11月西城分公司的移动新增用户是多少？",
    "history": [
      {"role": "user", "content": "这个表有哪些列？"},
      {"role": "assistant", "content": "该表包含以下列：账期、分公司名称、..."}
    ]
  }'
```

## 🏗️ 项目结构

```
Excel_Agent/
├── config.yaml              # 配置文件
├── pyproject.toml           # 项目依赖
├── README.md
├── knowledge/               # 知识库文件目录
│   └── *.md                 # Markdown 格式知识文件
├── .vector_db/              # Chroma 向量数据库（自动生成）
└── src/
    └── excel_agent/
        ├── __init__.py
        ├── main.py          # 入口
        ├── api.py           # FastAPI 接口
        ├── config.py        # 配置管理
        ├── excel_loader.py  # Excel 加载器
        ├── graph.py         # LangGraph 工作流
        ├── knowledge_base.py # 知识库管理
        ├── prompts.py       # 提示词模板
        ├── stream.py        # 流式对话核心
        ├── tools.py         # 数据分析工具
        └── frontend/
            └── index.html   # Web 界面
```

## 🐳 Docker 部署

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
EXPOSE 8000
CMD ["uv", "run", "python", "-m", "excel_agent.main", "serve"]
```

```bash
docker build -t excel-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key excel-agent
```

## 🔧 开发

```bash
# 安装开发依赖
uv sync --dev

# 运行测试（如有）
uv run pytest

# 代码格式化
uv run ruff format .
```

## 📝 使用示例

```
用户：这个表有多少行数据？
助手：该表共有 15,234 行数据。

用户：按分公司统计移动新增用户总数
助手：[调用 group_and_aggregate 工具]
      各分公司移动新增用户统计如下：
      | 分公司 | 移动新增用户 |
      |--------|-------------|
      | 东城   | 45,678      |
      | 西城   | 38,901      |
      | ...    | ...         |

用户：西城的明细呢？
助手：[理解上下文，调用 filter_data]
      西城分公司的详细数据如下：...

用户：用饼图展示各分公司的占比
助手：[调用 generate_chart 工具]
      📊 已生成饼图，共 8 个数据点。
      [交互式 ECharts 饼图显示]
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

[MIT License](LICENSE)

---

**Made with ❤️ using LangGraph, FastAPI, and OpenAI**
