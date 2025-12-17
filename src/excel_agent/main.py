"""主入口模块"""

import argparse
import sys

from .config import load_config, set_config


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="Excel 智能问数 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动 HTTP API 服务")
    serve_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="服务器主机地址",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="服务器端口",
    )
    
    # cli 命令
    cli_parser = subparsers.add_parser("cli", help="启动命令行交互模式")
    cli_parser.add_argument(
        "--excel", "-e",
        type=str,
        default=None,
        help="Excel 文件路径",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    set_config(config)
    
    if args.command == "serve":
        # 覆盖配置
        if args.host:
            config.server.host = args.host
        if args.port:
            config.server.port = args.port
        
        from .api import run_server
        print(f"🚀 启动 Excel Agent API 服务...")
        print(f"📍 地址: http://{config.server.host}:{config.server.port}")
        print(f"📚 API 文档: http://{config.server.host}:{config.server.port}/docs")
        run_server()
        
    elif args.command == "cli":
        run_cli(args.excel)
        
    else:
        # 默认启动服务
        from .api import run_server
        print(f"🚀 启动 Excel Agent API 服务...")
        print(f"📍 地址: http://{config.server.host}:{config.server.port}")
        print(f"📚 API 文档: http://{config.server.host}:{config.server.port}/docs")
        run_server()


def run_cli(excel_path: str = None):
    """运行命令行交互模式"""
    from langchain_core.messages import HumanMessage, AIMessage
    from .excel_loader import get_loader
    from .graph import get_graph
    
    print("=" * 50)
    print("📊 Excel 智能问数 Agent - CLI 模式")
    print("=" * 50)
    
    loader = get_loader()
    
    # 加载 Excel
    if excel_path:
        file_path = excel_path
    else:
        file_path = input("\n请输入 Excel 文件路径: ").strip()
    
    if not file_path:
        print("❌ 未提供文件路径，退出")
        return
    
    try:
        structure = loader.load(file_path)
        print(f"\n✅ 成功加载 Excel 文件!")
        print(f"📋 工作表: {structure['sheet_name']}")
        print(f"📏 数据规模: {structure['total_rows']} 行 × {structure['total_columns']} 列")
        print(f"\n📄 列信息:")
        for col in structure['columns']:
            print(f"   - {col['name']} ({col['dtype']})")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    graph = get_graph()
    
    print("\n" + "=" * 50)
    print("💬 开始对话 (输入 'exit' 或 'quit' 退出)")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("👋 再见!")
            break
        
        try:
            result = graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "is_relevant": True,
            })
            
            # 提取最后的 AI 响应
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 Agent: {msg.content}")
                    break
        except Exception as e:
            print(f"\n❌ 处理出错: {e}")


if __name__ == "__main__":
    main()
