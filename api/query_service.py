import json
import uuid
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, StreamingResponse

from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from utils.clients.mongo_history_utils import get_recent_messages, clear_history
from utils.path_util import PROJECT_ROOT
from common.logging.logger import logger
from utils.sse_utils import sse_generator, create_sse_queue, push_to_session, SSEEvent
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, \
    get_done_task_list, clear_task

app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 跳转到查询页面的接口
@app.get("/chat.html")
def chat():
    chat_html_path_obj = PROJECT_ROOT / "page" / "chat.html"
    if not chat_html_path_obj.exists():
        logger.error("chat.html页面不存在")
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path_obj}！")
    return FileResponse(chat_html_path_obj)

# 健康检查接口
@app.get("/health")
def health():
    logger.info(f"{datetime.now()}:完成本次健康状态检查!")
    return {"ok": True}

# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

# 执行查询图流程
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    logger.info(f"[{session_id}] 开始执行查询流程，流式模式：{is_stream}")
    try:
        # 更新全局状态为processing
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        # 定义状态，初始化数据
        state = create_query_default_state(
            session_id=session_id,
            is_stream=is_stream,
            original_query=user_query
        )
        # 调用图执行
        logger.info(f"查询图开始执行,传入参数:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        state = query_app.invoke(state)
        logger.info(f"查询图执行结束,返回结果:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        # 更新全局状态为processing
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        # 整个图流程执行结束之后，通过sse推送final类型消息
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": state.get("answer"),
                    "status": "completed",
                    "image_urls": state.get("image_urls")
                }
            )
        return state

    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        logger.exception(f"{user_query}:执行图流程报错!")
        push_to_session(session_id=session_id, event=SSEEvent.ERROR, data={
            "error": str(e)
        })

# 用户查询接口
@app.post("/query")
def query(query_request: QueryRequest,background_tasks: BackgroundTasks):
    user_query = query_request.query
    session_id = query_request.session_id if query_request.session_id else str(uuid.uuid4())
    is_stream = query_request.is_stream
    logger.info(f"用户发起提问: {user_query} 关联的session_id:{session_id},流式状态:{is_stream}")
    # 判断是否是流式输出
    if is_stream:
        # 执行后台任务

        # 清除当前session_id对应的状态数据
        clear_task(session_id)
        # 必须在返回session_id之前创建
        create_sse_queue(session_id)

        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        logger.info(f"[{session_id}] 流式任务已提交至后台执行")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        state = run_query_graph(session_id, user_query, is_stream)
        done_list = get_done_task_list(session_id)

        return {
            "session_id": session_id,
            "message": f"{session_id}已经完成问题解析!",
            "answer": state.get("answer"),
            "done_list": done_list,
            "image_urls": state.get("image_urls", [])
        }

# SE 流式推送接口
@app.get("/stream/{session_id}")
def stream(session_id: str,request : Request):
    logger.info(f"{datetime.now()}完成{session_id}对应的流式响应接口!")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )

# 历史数据处理--查询
@app.get("/history/{session_id}")
def get_history(session_id:str):
    try:
        # 根据session_id到mongodb中获取当前会话对应的所有的对话记录
        records = get_recent_messages(session_id)
        logger.info(f"查询{session_id}的聊天记录,查询到:{len(records)}条!")
        # 注意：直接从mongodb中查询出来的数据，主键是ObjectId类型，不能直接向前端返回，需要先进行一次转换
        items:list[dict] = []
        for record in records:
            items.append({
                "_id": str(record.get("_id")) if record.get("_id") is not None else "",
                "session_id": session_id,
                "role": record.get("role", ""),
                "text": record.get("text", ""),
                "rewritten_query": record.get("rewritten_query", ""),
                "item_names": record.get("item_names", []),
                "image_urls": record.get("image_urls", []),
                "ts": record.get("ts")
            })
        return {
            "session_id": session_id,
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

# 历史数据处理--删除
@app.delete("/history/{session_id}")
def clear_chat_history(session_id: str):
    # 根据session_id,将当前会话对应的历史数据删除掉
    deleted_count = clear_history(session_id)
    logger.info(f"已经清空:{session_id}对应的聊天记录,删除的数量为:{deleted_count}")
    return {"message": "History cleared", "deleted_count": deleted_count}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)


