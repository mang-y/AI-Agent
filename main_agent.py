"""
AI Agent 主服务（FastAPI）
==========================

将基于 LangGraph 的 Agent 工作流通过 FastAPI 暴露为 HTTP 接口。

核心改造点（相对于原 RAG 项目）：
1. **LCEL → LangGraph**：原 conversation_chain (LCEL) 替换为 agent_graph (LangGraph)
2. **固定流水线 → Agent 自主决策**：LLM 根据用户意图自主选择工具
3. **接入 LangSmith**：通过环境变量自动追踪全链路
4. **新增 Agent 工具**：rag_search / rag_answer / upload_document / evaluate_rag 等
5. **多会话管理**：新增 /sessions 系列接口，支持 ChatGPT 风格的多会话切换
6. **前端静态托管**：挂载 static/ 目录，直接通过浏览器访问

API 端点：
- POST /chat                       流式对话（SSE，token 级流式输出）
- POST /chat/sync                  同步对话（一次性返回完整结果）
- POST /upload                     上传文档到知识库
- GET  /sessions                   列出全部会话
- POST /sessions                   创建新会话
- GET  /sessions/{sid}/messages    获取指定会话的消息列表
- DELETE /sessions/{sid}           删除会话
- PUT  /sessions/{sid}             重命名会话
- GET  /history                    查询会话历史（兼容旧接口）
- DELETE /history                  清空会话历史（兼容旧接口）
- GET  /tools                      列出 Agent 可用的工具
- GET  /health                     健康检查
"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import logging
import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

# 加载环境变量（含 LangSmith 配置）
load_dotenv()

import config_data as config
from graph import agent_graph
from tools import ALL_TOOLS
from tools.rag_tools import rag_search, rag_answer
from tools.file_tools import upload_document, list_documents
from tools.history_tools import get_chat_history, clear_history
from file_history_store import (
    get_history,
    create_session,
    list_sessions,
    delete_session,
    rename_session,
    get_session_messages,
    auto_title_if_needed,
)
from rag_infra import hybrid_retrieve_with_rerank, format_document, rewrite_query

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# LangSmith 配置检查
# ============================================================
def _check_langsmith_config():
    """检查 LangSmith 是否已正确配置"""
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGSMITH_PROJECT", config.LANGSMITH_PROJECT)

    if tracing and api_key:
        logger.info(f"✅ LangSmith 追踪已启用 | 项目: {project}")
        logger.info(f"   查看追踪: https://smith.langchain.com/o/default/projects/p/{project}")
    elif tracing and not api_key:
        logger.warning("⚠️ LANGSMITH_TRACING=true 但未设置 LANGSMITH_API_KEY，追踪将不生效")
    else:
        logger.info("ℹ️ LangSmith 追踪未启用（设置 LANGSMITH_TRACING=true 开启）")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="AI Agent Service (LangGraph + LangSmith)",
    description="基于 LangGraph 的智能 Agent 服务，集成 RAG、文档管理、评估等工具能力，支持多会话与流式输出",
    version="2.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（前端页面）
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"静态文件目录已挂载: {STATIC_DIR}")


# ============================================================
# 请求/响应模型
# ============================================================
class ChatRequest(BaseModel):
    input: str
    session_id: Optional[str] = "default_session"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    iterations: int


class UploadTextRequest(BaseModel):
    content: str
    source_name: str = "api_upload"


class CreateSessionRequest(BaseModel):
    title: str = ""


class RenameSessionRequest(BaseModel):
    title: str


# ============================================================
# 前端页面入口
# ============================================================
@app.get("/")
async def index():
    """返回前端聊天页面"""
    index_html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    raise HTTPException(status_code=404, detail="前端页面未找到，请检查 static/index.html 是否存在")


# ============================================================
# 健康检查
# ============================================================
@app.get("/health")
async def health():
    """健康检查 + 服务信息"""
    return {
        "status": "ok",
        "service": "AI Agent (LangGraph + LangSmith)",
        "version": "2.1.0",
        "langsmith_enabled": os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", config.LANGSMITH_PROJECT),
        "tools_count": len(ALL_TOOLS),
        "tools": [t.name for t in ALL_TOOLS],
    }


# ============================================================
# 工具列表
# ============================================================
@app.get("/tools")
async def list_tools():
    """列出 Agent 可用的所有工具及其描述"""
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "args_schema": t.args_schema.model_json_schema() if t.args_schema else None,
            }
            for t in ALL_TOOLS
        ]
    }


# ============================================================
# 会话管理接口（多会话核心）
# ============================================================
@app.get("/sessions")
async def api_list_sessions():
    """列出全部会话（按更新时间倒序）"""
    try:
        sessions = list_sessions()
        return JSONResponse(content={"sessions": sessions, "total": len(sessions)})
    except Exception as e:
        logger.error(f"列出会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions")
async def api_create_session(req: CreateSessionRequest):
    """创建新会话，返回 session_id"""
    try:
        session = create_session(title=req.title)
        logger.info(f"创建会话: {session['session_id']}")
        return JSONResponse(content=session)
    except Exception as e:
        logger.error(f"创建会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages")
async def api_get_session_messages(session_id: str, limit: int = 100):
    """获取指定会话的消息列表"""
    try:
        messages = get_session_messages(session_id, limit=limit)
        return JSONResponse(content={
            "session_id": session_id,
            "messages": messages,
            "count": len(messages),
        })
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """删除指定会话"""
    try:
        delete_session(session_id)
        logger.info(f"删除会话: {session_id}")
        return JSONResponse(content={"status": "success", "session_id": session_id})
    except Exception as e:
        logger.error(f"删除会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/sessions/{session_id}")
async def api_rename_session(session_id: str, req: RenameSessionRequest):
    """重命名会话"""
    try:
        result = rename_session(session_id, req.title)
        logger.info(f"重命名会话 {session_id} → {req.title}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"重命名会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 对话接口（同步）
# ============================================================
@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(req: ChatRequest):
    """
    同步对话接口：一次性返回 Agent 的最终回答。

    Agent 会自主决策是否调用工具、调用哪些工具，
    整个过程在 LangSmith 中可追踪。
    """
    if not req.input:
        raise HTTPException(status_code=400, detail="input 字段不能为空")

    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"

    initial_state = {
        "messages": [HumanMessage(content=req.input)],
        "session_id": session_id,
        "iteration": 0,
        "retrieved_docs": None,
        "metadata": {"session_id": session_id, "source": "api_sync"},
    }

    runnable_config = RunnableConfig(
        configurable={"session_id": session_id},
        metadata={
            "session_id": session_id,
            "user_input": req.input[:100],
            "interface": "api_sync",
        },
        tags=["agent", "rag-agent"],
    )

    try:
        final_state = agent_graph.invoke(initial_state, config=runnable_config)

        last_msg = final_state["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        iterations = final_state.get("iteration", 0)

        # 保存到历史记录
        history = get_history(session_id)
        history.add_messages([
            HumanMessage(content=req.input),
            AIMessage(content=answer),
        ])
        # 自动生成会话标题
        auto_title_if_needed(session_id)

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            iterations=iterations,
        )
    except Exception as e:
        import traceback
        logger.error(f"对话失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")


# ============================================================
# 对话接口（流式 SSE）—— 前端多会话页面的核心
# ============================================================
@app.post("/chat")
async def chat_stream(request: Request, req: ChatRequest):
    """
    流式对话接口（SSE）：实时推送 Agent 的输出 token。

    事件类型：
    - token:      LLM 输出的 token 片段
    - tool_start: 工具调用开始
    - tool_end:   工具调用结束
    - done:       流式结束
    - error:      错误信息
    """
    if not req.input:
        raise HTTPException(status_code=400, detail="input 字段不能为空")

    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"

    initial_state = {
        "messages": [HumanMessage(content=req.input)],
        "session_id": session_id,
        "iteration": 0,
        "retrieved_docs": None,
        "metadata": {"session_id": session_id, "source": "api_stream"},
    }

    runnable_config = RunnableConfig(
        configurable={"session_id": session_id},
        metadata={
            "session_id": session_id,
            "user_input": req.input[:100],
            "interface": "api_stream",
        },
        tags=["agent", "rag-agent", "stream"],
    )

    async def event_generator():
        full_answer_parts = []
        try:
            # 推送会话 ID，便于前端确认
            yield {"event": "session", "data": session_id}

            async for event in agent_graph.astream_events(
                initial_state,
                config=runnable_config,
                version="v2",
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")

                # 1. LLM 流式 token
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        # 某些模型返回结构化 content（如 [{"text": "..."}]）
                        if isinstance(content, list):
                            extracted = ""
                            for part in content:
                                if isinstance(part, dict) and "text" in part:
                                    extracted += part["text"]
                                elif isinstance(part, str):
                                    extracted += part
                            content = extracted
                        if isinstance(content, str) and content:
                            full_answer_parts.append(content)
                            yield {"event": "token", "data": content}

                # 2. 工具调用开始
                elif event_type == "on_tool_start":
                    tool_input = event.get("data", {}).get("input", {})
                    yield {
                        "event": "tool_start",
                        "data": json_dumps_safe({
                            "tool": event_name,
                            "input": str(tool_input)[:200],
                        }),
                    }

                # 3. 工具调用结束
                elif event_type == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    yield {
                        "event": "tool_end",
                        "data": json_dumps_safe({
                            "tool": event_name,
                            "output": str(output)[:300],
                        }),
                    }

            # 流结束后保存历史
            full_answer = "".join(full_answer_parts)
            if full_answer:
                history = get_history(session_id)
                history.add_messages([
                    HumanMessage(content=req.input),
                    AIMessage(content=full_answer),
                ])
                # 自动生成会话标题
                auto_title_if_needed(session_id)

            yield {"event": "done", "data": "[DONE]"}
        except Exception as e:
            import traceback
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"流式对话失败: {err_msg}")
            yield {"event": "error", "data": err_msg}

    return EventSourceResponse(event_generator())


def json_dumps_safe(obj) -> str:
    """安全的 JSON 序列化（用于 SSE data 字段）"""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# ============================================================
# 文档上传接口
# ============================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件到知识库（自动分块 + 向量化）"""
    try:
        content = (await file.read()).decode("utf-8", errors="ignore")
        if not content.strip():
            raise HTTPException(status_code=400, detail="文件内容为空")

        result = upload_document.invoke({
            "content": content,
            "source_name": file.filename or "file_upload",
        })
        return JSONResponse(content={"status": "success", "result": result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/text")
async def upload_text(req: UploadTextRequest):
    """通过文本内容直接上传到知识库"""
    try:
        result = upload_document.invoke({
            "content": req.content,
            "source_name": req.source_name,
        })
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 历史记录接口（兼容旧版）
# ============================================================
@app.get("/history")
async def get_history_api(session_id: str = "default_session", limit: int = 20):
    """获取会话历史记录"""
    try:
        result = get_chat_history.invoke({
            "session_id": session_id,
            "limit": limit,
        })
        return JSONResponse(content={"session_id": session_id, "history": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history")
async def clear_history_api(session_id: str = "default_session"):
    """清空会话历史记录"""
    try:
        result = clear_history.invoke({"session_id": session_id})
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 文档列表接口
# ============================================================
@app.get("/documents")
async def list_docs(limit: int = 50):
    """列出知识库中的文档"""
    try:
        result = list_documents.invoke({"limit": limit})
        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 启动
# ============================================================
@app.on_event("startup")
async def startup_event():
    """启动时检查配置"""
    _check_langsmith_config()
    logger.info(f"🚀 AI Agent 服务启动完成")
    logger.info(f"   工具数量: {len(ALL_TOOLS)}")
    logger.info(f"   工具列表: {[t.name for t in ALL_TOOLS]}")
    logger.info(f"   前端页面: http://localhost:8000/")
    logger.info(f"   API 文档: http://localhost:8000/docs")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
