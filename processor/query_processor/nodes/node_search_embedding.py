import time
import sys

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import add_done_task,add_running_task


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取item_names
    # 从状态中获取rewritten_query
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")

    # 合法性校验
    if (not item_names) or (not rewritten_query):
        logger.warning(f"item_names或者rewritten_query为空，item_names={item_names}, rewritten_query={rewritten_query}")
        raise ValueError(f"item_names或者rewritten_query为空,业务无法继续,提前终止!")

    return item_names, rewritten_query


@step_log("step_2_select_chunks_in_milvus")
def step_2_select_chunks_in_milvus(item_names:list[str], rewritten_query):
    """
        向量化搜索
    :param item_names:
    :param rewritten_query:
    :return:   real_response  list[{id:xx,distance:xxx,entity:{}},{},{}]
    """
    # 生成向量数据
    result = generate_embeddings([rewritten_query])
    dense_vector = result.get("dense")[0]
    sparse_vector = result.get("sparse")[0]

    # 拼接过滤条件
    expr = f"item_name in {item_names}"
    # 封装AnnSearchRequest列表
    reqs_list = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr
    )

    # 执行混合检索
    response = hybrid_search(
        client = get_milvus_client(),
        collection_name= milvus_config.chunks_collection,
        reqs= reqs_list,
        ranker_weights=(0.5,0.5),
        norm_score=True,
        output_fields=[
            "chunk_id",
            "content",
            "title",
            "parent_title",
            "file_title",
            "item_name",
            "part"
        ]
    )

    if not response:
        return []

    return response[0]

@step_log("step_3_after_deal_milvus_result")
def step_3_after_deal_milvus_result(real_response):
    """
        将向量化的查询结果进行格式化 [{id:xx,distance:xxx,entity:{"chunk_id":xxx,"content":xx...}},{},{}]
    :param real_response:
    :return:
    """
    # 用于接收格式化的结果
    embedding_chunks: list[dict] = []
    if real_response:
        for item in real_response:
            chunk = item.get("entity")
            chunk["score"] = item.get("distance")
            chunk["type"] = "milvus"
            embedding_chunks.append(chunk)
    logger.info(f"完成了问题向量检索!检索的数量:{len(embedding_chunks)}")
    return embedding_chunks


@node_log("node_search_embedding")
def node_search_embedding(state:QueryGraphState):
    """
    节点功能：进行向量内容检索
    """
    # TODO 1. 将节点加入到运行时列表
    add_running_task(state.get("session_id"),sys._getframe().f_code.co_name,state.get("is_stream"))
    # TODO 2. 从状态中获取数据并进行校验
    item_names, rewritten_query = step_1_validate_and_get_data(state)
    # TODO 3. 向量化搜索
    real_response: list[dict] = step_2_select_chunks_in_milvus(item_names, rewritten_query)
    # TODO 4. 格式化数据
    embedding_chunks = step_3_after_deal_milvus_result(real_response)
    # TODO 5. 将节点加入到已完成列表
    add_done_task(state.get("session_id"),sys._getframe().f_code.co_name,state.get("is_stream"))
    return {"embedding_chunks":embedding_chunks}

if __name__ == "__main__":
    # 模拟测试数据，注意：虽然函数注解是QueryGraphState，测试直接传dict也可以
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "电源线 请勿将设备连接到受墙壁开关或自",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],              # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 【扁平化业务字段，不再读取entity嵌套】
            print(f"chunk_id: {top1.get('chunk_id')}")
            print(f"score(相似度): {top1.get('score')}")
            print(f"type: {top1.get('type')}")
            print(f"Item Name: {top1.get('item_name')}")
            print(f"file_title: {top1.get('file_title')}")
            print(f"Content Preview: {top1.get('content', '')[:120]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查：")
            print("   1.Milvus集合是否存在对应item_name='HAK 180 烫金机'的数据")
            print("   2.环境变量 CHUNKS_COLLECTION 是否配置正确")
            print("   3.filter expr表达式日志，确认过滤条件是否符合预期")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
# import sys
# from common.logging.logger import node_log
# from processor.query_processor.state import QueryGraphState
# from utils.task_utils import add_running_task, add_done_task
#
#
# @node_log("node_search_embedding")
# def node_search_embedding(state:QueryGraphState):
#     add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
#     add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
#     return {"embedding_chunks":{}}