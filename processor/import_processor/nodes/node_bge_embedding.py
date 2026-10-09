import json
import sys
from pathlib import Path
from typing import Any

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import add_running_task, add_done_task

# 向量化批次大小：每批处理 5 条切片，避免显存溢出
EMBEDDING_BATCH_SIZE = 5

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取chunks
    chunks = state.get("chunks")
    # 如果没有从状态中获取到chunks
    if not chunks:
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        if md_path_obj.is_file():
            json_path_obj:Path = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                # 从json备份文件中读取数据
                chunks:list[dict[str,Any]] = json.loads(json_path_obj.read_text(encoding="utf-8"))
                for chunk in chunks:
                    chunk["item_name"] = state.get("item_name")
                state["chunks"] = chunks
            else:
                logger.error(f"chunks没有值,json备份文件不存在,抛出异常!")
                raise ValueError(f"chunks没有值,json备份文件不存在,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到对应md_path数据,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应md_path数据,抛出异常!")
    return chunks

@step_log("step_2_batch_generate_vector")
def step_2_batch_generate_vector(chunks):
    #  chunks [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16...]
    # 希望5条5条的从chunks取chunk [1,2,3,4,5] [6,7,8,9,10],[11,12,13,14,15],[16...]
    # chunks[start:end]   ===> chunk{"parent_title":"","part":"","title":"","file_title":"","item_name":"","content":"",}
    # 注意：攒批的意义->在做嵌入操作的时候，每5条数据和嵌入模型交互一次，减少交互次数
    for i in range(0,len(chunks),EMBEDDING_BATCH_SIZE):
        current_chunks = chunks[i:i+EMBEDDING_BATCH_SIZE]
        content_list:list[str] = [chunk.get("item_name") + "_" + chunk.get("content") for chunk in current_chunks]

        # 调用嵌入模型，给列表中的每一条文本进行嵌入
        result = generate_embeddings(content_list)
        dense_list = result["dense"]
        sparse_list = result["sparse"]
        for index,chunk in enumerate(current_chunks) :
            chunk["dense_vector"] = dense_list[index]
            chunk["sparse_vector"] = sparse_list[index]

    logger.info(f"chunks已经批量生成向量! {chunks[:2]}")


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    """
    # TODO 1. 添加运行时状态
    add_running_task(state['task_id'], "node_bge_embedding")
    # TODO 2. 从状态中获取数据并进行校验
    chunks = step_1_validate_and_get_data(state)
    # TODO 3. 批量向量化
    step_2_batch_generate_vector(chunks)
    # TODO 4. 更新状态
    state["embeddings_content"] = chunks
    # TODO 5. 添加结束状态
    add_done_task(state['task_id'], "node_bge_embedding")
    return state

if __name__ == '__main__':
    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游商品名称识别节点产出）
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        logger.info(f"返回的结果:{result_chunks}")


    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")