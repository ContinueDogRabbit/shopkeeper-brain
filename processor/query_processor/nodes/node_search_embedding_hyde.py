import time
import sys

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.config.milvus_config import milvus_config
from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import  add_done_task,add_running_task

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state:QueryGraphState):
    # 1. 获取数据
    item_names:list[str] = state.get("item_names",[])
    rewritten_query:str  = state.get("rewritten_query")
    # 2. 参数校验
    if (not item_names) or (not rewritten_query):
        # [ for item in list if xx  else xxx -> 没有]
        # 值  if  xx else 值
        logger.error(f"item_names或者rewritten_query为空,业务无法继续,提前终止!")
        raise ValueError(f"item_names或者rewritten_query为空,业务无法继续,提前终止!")
    # 3. 返回结果
    return item_names,rewritten_query


@step_log("step_2_call_llm_by_rewritten_query")
def step_2_call_llm_by_rewritten_query(rewritten_query):
    # 获取大语言模型对象
    llm_model = get_llm_client()
    # 拼接提示词
    prompt_text = load_prompt("hyde_prompt",rewritten_query = rewritten_query)
    # 封装消息
    message = HumanMessage(
        content=prompt_text
    )

    # 创建调用链
    chains = llm_model | StrOutputParser()
    # 调用获取结果
    hyde_answer = chains.invoke([message])
    # 返回
    logger.info(f"基于:{rewritten_query},模型给与的假设性回答:{hyde_answer}")
    return hyde_answer


@step_log("step_3_select_chunks_in_milvus")
def step_3_select_chunks_in_milvus(item_names, rewritten_query, hyde_answer):
    """
    向量化查询
    :param item_names:        用于过滤商品，减法搜索范围
    :param rewritten_query:   重写的问题
    :param hyde_answer:       假设性答案
    :return:     list[dict]     [{id:xxx,distance:xxxx,entity:{}},{},{}]
    """
    # 向量化处理
    result = generate_embeddings([f"问题：{rewritten_query}，假设性答案：{hyde_answer}"])
    dense_vector =  result.get("dense")[0]
    sparse_vector = result.get("sparse")[0]

    # 封装AnnSearchRequest列表
    reqs_list = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr= f"item_name in {item_names}"
    )

    # 执行混合检索
    response = hybrid_search(
        client = get_milvus_client(),
        collection_name=milvus_config.chunks_collection,
        reqs = reqs_list,
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


@step_log("step_4_after_deal_milvus_result")
def step_4_after_deal_milvus_result(real_response):
    hyde_embedding_chunks: list[dict] = []
    # real_response :    [{id:xxx,distance:xxxx,entity:{}},{},{}]
    if real_response:
        for item in real_response:
            chunk = item.get("entity")
            chunk["score"] = item.get("distance")
            chunk["type"] = "milvus"
            hyde_embedding_chunks.append(chunk)
    logger.info(f"完成了问题向量检索!检索的数量:{len(hyde_embedding_chunks)}")
    return hyde_embedding_chunks


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state:QueryGraphState):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # TODO 2. 从状态中获取数据并进行校验
    item_names, rewritten_query = step_1_validate_and_get_data(state)
    # TODO 3. 调用模型生成假设性答案
    hyde_answer = step_2_call_llm_by_rewritten_query(rewritten_query)
    # TODO 4. 向量化查询
    real_response: list[dict] = step_3_select_chunks_in_milvus(item_names, rewritten_query, hyde_answer)
    # TODO 5. 格式化输出
    hyde_embedding_chunks = step_4_after_deal_milvus_result(real_response)
    # TODO 6. 添加节点到已完成列表
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks":hyde_embedding_chunks}

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }
    result = node_search_embedding_hyde(mock_state)
    print(result)
# import sys
# from common.logging.logger import node_log
# from processor.query_processor.state import QueryGraphState
# from utils.task_utils import add_running_task, add_done_task
#
#
# @node_log("node_search_embedding_hyde")
# def node_search_embedding_hyde(state:QueryGraphState):
#     add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
#     add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
#     return {"hyde_embedding_chunks":{}}