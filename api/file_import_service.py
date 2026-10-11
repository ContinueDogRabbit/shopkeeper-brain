"""
    该模块是导入的接口处理模块
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import FileResponse
from common.logging.logger import logger
from processor.import_processor.main_graph import kb_import_app
from processor.import_processor.state import get_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import get_running_task_list, get_done_task_list, get_task_status, add_running_task, \
    add_done_task, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, TASK_STATUS_COMPLETED

# 创建fastapi应用对象
app = FastAPI()

# 配置cors跨域
# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 页面访问的接口
@app.get("/import.html")
async def get_import_page():
    html_abs_path = PROJECT_ROOT / "page"/ "import.html"

    # 以FileResponse返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # 显式指定媒体类型为HTML，确保浏览器正确解析
    )
# 导入图调用--后台任务
def run_graph_task(task_id:str,local_dir:str,local_file_path:str):
    try:
        # 更新任务全局状态为：处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        for event in kb_import_app.stream(init_state,stream_mode="updates"):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")
    except Exception as e:
        # 更新任务全局状态为：处理中
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)


# 上传文件接口
@app.post("/upload")
async def upload_file(files: list[UploadFile],background_tasks: BackgroundTasks):
    # 构建存储路径
    # 获取日期字符串
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    task_ids = []
    for file in files:
        # 生成不重复的随机数，作为task_id
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        # 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 构建上传文件的本地保存绝对路径
        local_file_abs_path: Path = task_local_dir / file.filename

        # 文件保存到磁盘操作
        data = await file.read()
        local_file_abs_path.write_bytes(data)

        # 标记「文件上传」阶段为「已完成」
        add_done_task(task_id, "upload_file")

        # 使用后台任务 调用图流程对文件进行处理 pdf/md---->milvus数据库
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path)
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")
    # 所有文件处理完毕，返回上传成功信息和所有TaskID
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }


# 查询状态接口
@app.get("/status/{task_id}")
async def get_task_progress(task_id: str):
    # 根据task_id 获取运行时节点、已完成节点、图整体状态
    running_tasks = get_running_task_list(task_id)
    done_tasks =  get_done_task_list(task_id)
    task_status =  get_task_status(task_id)

    task_status_info: dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": task_status,  # 任务全局状态：pending/processing/completed/failed
        "done_list":  done_tasks,  # 已完成的节点/阶段列表
        "running_list": running_tasks  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info



if __name__ == "__main__":
    # uvicorn.run("file_import_service:app", host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)