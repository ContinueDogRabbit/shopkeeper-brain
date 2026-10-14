# import time
# import sys
#
# from processor.query_processor.nodes.node_search_embedding import node_search_embedding
# from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
# from processor.query_processor.state import QueryGraphState
# from utils.task_utils import add_running_task, add_done_task
# from common.logging.logger import logger, node_log, step_log
#
#
# @step_log("step_1_validate_and_get_data")
# def step_1_validate_and_get_data(state):
#     """
#       获取核心参数并且校验
#     :param state:
#     :return:
#     """
#     #1. 获取参数
#     embedding_chunks = state.get("embedding_chunks")
#     hyde_embedding_chunks = state.get("hyde_embedding_chunks")
#     #2. 非空校验
#     if (not embedding_chunks) or (not hyde_embedding_chunks):
#         logger.error(f"embedding_chunks或者hyde_embedding_chunks为空,业务无法继续,提前终止!")
#         raise ValueError(f"embedding_chunks或者hyde_embedding_chunks为空,业务无法继续,提前终止!")
#     #3. 返回结果
#     return embedding_chunks,hyde_embedding_chunks
#
#
# @step_log("step_2_use_rrf_rank")
# def step_2_use_rrf_rank(data_list,k:int = 60):
#     # data_list : [(0.6,embedding_chunks),(0.4,hyde_embedding_chunks)]
#     # embedding_chunks|hyde_embedding_chunks   [{"chunk_id":xx,"content":xx,"file_tile":...},{},{}]
#     # 用于存放最后返回的结果的列表
#     rrf_chunks = []
#
#     # 定义一个字典，用于存储当前chunk的分数  k:chunk_id  v:score
#     socre_dict:dict = {}
#
#     # 定义一个字典，用于存储当前chunk       k:chunk_id  v:chunk
#     chunk_dict:dict ={}
#
#     # 对2路结果进行遍历  weight：当前这路结果的权重，chunk_list 当前这路的查询结果列表
#     for weight,chunk_list in data_list:
#         # 对当前这里的查询结果列表进行遍历
#         for rank,chunk in enumerate(chunk_list,start=1):
#             # 计算这路的这个chunk的分值
#             score = weight * 1/(k+rank)
#             # 获取chunk_id
#             chunk_id = chunk.get("chunk_id")
#             # 将当前chunk对应的得分放到score_dict中
#             socre_dict[chunk_id] = socre_dict.get(chunk_id,0) + score
#             # 将当前chunk放到chunk_dict中   如果同一个chunk在多路中出现多次，去重
#             chunk_dict[chunk_id] = chunk
#
#     # 对chunk_dict进行遍历
#     for chunk_id,chunk in chunk_dict.items():
#         chunk["score"] = socre_dict.get(chunk_id)
#         rrf_chunks.append(chunk)
#
#     # rrf_chunks进行排名
#     logger.debug(f"排序之前,长度:{len(rrf_chunks)},内容: {rrf_chunks}")
#     # 指定按照分值进行降序排序
#     rrf_chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
#     logger.debug(f"排序之后,长度:{len(rrf_chunks)},内容: {rrf_chunks}")
#
#     rrf_chunks = rrf_chunks[:5]
#
#     return rrf_chunks
#
#
# @node_log("node_rrf")
# def node_rrf(state:QueryGraphState):
#     """
#     节点功能：Reciprocal Rank Fusion
#     将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
#     """
#     # TODO 1. 添加节点到运行时列表
#     add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
#     # TODO 2. 从状态中获取数据并进行校验
#     embedding_chunks, hyde_embedding_chunks = step_1_validate_and_get_data(state)
#     # TODO 3. 将两路查询结果封装为一个元组，放到大的列表中
#     # [(0.6,embedding_chunks),(0.4,hyde_embedding_chunks)]
#     # embedding_chunks|hyde_embedding_chunks   [{"chunk_id":xx,"content":xx,"file_tile":...},{},{}]
#     data_list = [
#         (0.5,embedding_chunks),
#         (0.5,hyde_embedding_chunks)
#     ]
#     # TODO 4. 自己实现rrf + 权重融合排序
#     rrf_chunks = step_2_use_rrf_rank(data_list)
#     state["rrf_chunks"] = rrf_chunks
#     # TODO 5. 添加节点到已完成列表
#     add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
#     return state
#
# if __name__ == "__main__":
#     mock_state = {
#         "session_id": "test_rrf_session",
#         "is_stream": False,
#         "original_query": "HAK 180 烫金机怎么操作？",
#         "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
#         "item_names": ["HAK 180 烫金机"],
#     }
#
#     emb_res = node_search_embedding(mock_state)
#     hyde_res = node_search_embedding_hyde(mock_state)
#     mock_state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
#     mock_state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []
#
#     result = node_rrf(mock_state)
#     print(result)
import sys
from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task


@node_log("node_rrf")
def node_rrf(state:QueryGraphState):
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
    return state