import sys

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task, add_done_task
from pathlib import Path

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    """
    # TODO 添加到运行中列表
    add_running_task(state.get("task_id"),"node_entry")

    # TODO 从状态中获取local_file_path
    local_file_path = state.get("local_file_path")

    if not local_file_path:
        logger.warning("没有指定上传文件的路径")
        return state

    # TODO 判断文件类型
    if local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False

        state["md_path"] = local_file_path
        state["pdf_path"] = None
    elif local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False

        state["pdf_path"] = local_file_path
        state["md_path"] = None
    else:
        logger.warning(f" 不支持的文件类型: {local_file_path}，终止流程")
        return state
    # D:\dev\workspace\shopkeeper - brain - 0615\doc\hak180产品安全手册.pdf
    # TODO 补充file_title
    # state["file_title"] = local_file_path.split("/")[-1].split(".")[0]
    # state["file_title"] = os.path.basename(local_file_path).split(".")[0]
    # .name 获取文件名(带后缀)    .stem 获取文件名(不带后缀)   .parent 获取当前文件所在的文件夹名   .suffix 获取后缀
    state["file_title"] = Path(local_file_path).stem

    # TODO 添加到已完成列表
    add_done_task(state.get("task_id"),"node_entry")
    return state