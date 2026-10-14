# import re
# import time
# import sys
# from typing import Any
#
# from common.logging.logger import logger, node_log, step_log
# from processor.query_processor.state import QueryGraphState
# from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
# from utils.lm.lm_utils import get_llm_client
# from utils.load_prompt import load_prompt
# from utils.sse_utils import push_to_session, SSEEvent
# from utils.task_utils import add_running_task, add_done_task
#
#
# @step_log("step_1_has_answer")
# def step_1_has_answer(state):
#     answer = state["answer"]
#     return True if answer else False
#
# @step_log("step_2_validate_and_get_data")
# def step_2_validate_and_get_data(state:QueryGraphState):
#     reranked_docs = state.get("reranked_docs")
#     session_id = state.get("session_id")
#     item_names = state.get("item_names")
#     rewritten_query = state.get("rewritten_query")
#     is_stream = state.get("is_stream",False)
#
#     if (not reranked_docs) or (not session_id) or (not item_names) or (not rewritten_query):
#         logger.error("核心参数为空,业务无法继续,提前终止!")
#         raise ValueError("核心参数为空,业务无法继续,提前终止!")
#     return reranked_docs,session_id,item_names,rewritten_query,is_stream
#
#
# @step_log("step_3_get_history_by_session_id")
# def step_3_get_history_by_session_id(session_id):
#     # 根据session_id到mongodb中获取当前会话对应的聊天记录
#     history_list: list[dict[str, Any]] = get_recent_messages(session_id)
#     logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
#     # 过滤出有效的历史记录     有效:记录中的item_name不能为空
#     history_list = [item for item in history_list if item.get("item_names")]
#     logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
#     return history_list
#
# @step_log("step_4_create_answer_prompt")
# def step_4_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query):
#     # 拼接context   从reranked_docs获取  [{"title":"","text":"","score":"","type":"","url":"",},{},{}]
#     # f"标题:xxx,数据类源:{'联网搜索' or '向量数据库'},置信度:xxx,内容:xxx
#     context :str = ""
#     for chunk in reranked_docs:
#         context +=  (f"标题:{chunk.get('title')},数据类源:{'联网搜索' if chunk.get('type')=='web' else '向量数据库'},"
#                      f"置信度:{chunk.get('score')},内容:{chunk.get('text')}\n")
#
#     if history_list:
#         # 在mongodb中有当前会话对应的历史记录
#         history_text_list :list[dict[str,Any]] = []
#         # 对历史记录进行遍历
#         for index,history_item in enumerate(history_list,start=1):
#             # 判断当前取出的这条历史记录是用户角色还是助手角色
#             if history_item.get("role") == "user":
#                 # 用户角色  编号:xxx,用户提问,原始问题:xx,重写后的问题:xxx，关联item_name：xxx
#                 history_text_list.append(f"编号:{index},用户提问,原始问题:{history_item.get('text')},"
#                                          f"重写后的问题:{history_item.get('rewritten_query')}，"
#                                          f"关联item_name：{','.join(history_item.get('item_names'))}")
#             else:
#                 # 助手角色  编号:xxx,助手回答,重写问题:xx,回答结果:xx,关联item_name:xxx"
#                 # 注意：助手回答内容可能比较多，为了避免内容过多可以使用如下方式进行优化
#                 # 方式1：对历史记录中的回答内容进行截取
#                 # 方式2： 调整提示词模版的内容
#                 history_text_list.append(f"编号:{index},助手回答,重写问题:{history_item.get('rewritten_query')},"
#                                          f"回答结果:{history_item.get('text')[:50]},关联item_name:{','.join(history_item.get('item_names'))}")
#         history_text = "\n".join(history_text_list)
#     else:
#         # 在mongodb中没有当前会话对应的历史记录
#         history_text = "没有有效的历史聊天记录!"
#
#     item_names_str = ",".join(item_names)
#
#     prompt_text = load_prompt("answer_out",context = context,history = history_text,item_names = item_names_str,question = rewritten_query)
#     return prompt_text
#
#
# @step_log("step_5_call_llm_create_answer")
# def step_5_call_llm_create_answer(answer_prompt, is_stream, session_id):
#
#     answer : str = ""
#     # 获取模型客户端
#     llm_model = get_llm_client()
#
#     # 判断是不是流式处理
#     if is_stream:
#         # 流式处理，需要调用模型的stream方法来获取结果，并将结果通过SSE方式  推到客户端
#         stream_res = llm_model.stream(answer_prompt)
#         for chunk in stream_res:
#             chunk_str = chunk.content
#             # 将流式输出的内容 推到队列
#             push_to_session(
#                 session_id=session_id,
#                 event=SSEEvent.DELTA,
#                 data={"delta":chunk_str}
#             )
#             answer += chunk_str
#     else:
#         # 非流式处理 直接调用模型的invoke方法，来获取结果
#         reponse = llm_model.invoke(answer_prompt)
#         answer = reponse.content
#     return answer
#
#
# @step_log("step_6_extract_chunk_and_url_image")
# def step_6_extract_chunk_and_url_image(reranked_docs):
#     # 从rerank精排过滤后的内容中提取图片出来
#     image_urls: list[str] = []
#
#     # 定义匹配图片的正则表达式
#     rep = re.compile(r"\!\[.*?\]\((.*?)\)")
#     # 对结果进行遍历
#     for chunk in reranked_docs:
#         url:str = chunk.get("url")
#         text = chunk.get("text")
#
#         if url:
#             if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
#                 image_urls.append(url)
#         if text:
#             image_url_list:list[str] = rep.findall(text)
#             if image_url_list:
#                 image_urls.extend(image_url_list)
#     logger.info(f"完成切片中的图片提取,提取数量:{len(image_urls)},内容:{image_urls}")
#     return image_urls
#
#
# @step_log("step_7_save_assistant_message")
# def step_7_save_assistant_message(state):
#     # 保存结果到mongodb
#     save_chat_message(
#         session_id= state.get("session_id"),
#         role = "assistant",
#         text = state.get("answer"),
#         rewritten_query= state.get("rewritten_query"),
#         item_names= state.get("item_names"),
#         image_urls= state.get("image_urls")
#     )
#
#
# @node_log("node_answer_output")
# def node_answer_output(state:QueryGraphState):
#     """
#     节点功能：进行过处理可以是流式输出可以整体输出！
#     """
#     # TODO 1. 添加节点到运行时列表
#     add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
#     # TODO 2. 从状态中获取答案
#     has_answer = step_1_has_answer(state)
#     if not has_answer:
#         # 状态中没有答案
#         # TODO 3. 从状态中获取数据，并进行校验
#         reranked_docs, session_id, item_names, rewritten_query, is_stream = step_2_validate_and_get_data(state)
#         # TODO 4. 根据session_id获取历史对话
#         history_list = step_3_get_history_by_session_id(session_id)
#         # TODO 5. 拼接提示词
#         answer_prompt: str = step_4_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query)
#         # TODO 6. 调用模型获取answer   注意：如果是流式处理(is_stream=True)  ，需要调用push_to_session将实时生成流式结果放到队列中
#         answer: str = step_5_call_llm_create_answer(answer_prompt, is_stream, session_id)
#         # TODO 7. 提取内容中的图片地址
#         image_urls: list[str] = step_6_extract_chunk_and_url_image(reranked_docs)
#         # TODO 8. 更新状态
#         state["answer"] = answer
#         state["image_urls"] = image_urls
#
#     # TODO 9. 保存结果到mongodb
#     step_7_save_assistant_message(state)
#     # TODO 10. 添加节点到已完成列表
#     add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
#
#     return state
#
# if __name__ == "__main__":
#     mock_reranked_docs = [
#         {
#             "chunk_id": "local_101",
#             "type": "milvus",
#             "title": "HAK 180 烫金机操作手册_v2.pdf",
#             "score": 0.95,
#             "text": """
#             HAK 180 烫金机的操作面板位于机器正前方。
#             具体的操作面板布局请参考下图：
#             ![操作面板布局图](http://local-server/images/panel_view.jpg)
#             """,
#         }
#     ]
#     mock_history = [
#         {"role": "user", "text": "你好，这款机器怎么用？", "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法"},
#     ]
#     mock_state = {
#         "session_id": "test_answer_session_001",
#         "original_query": "HAK 180 烫金机怎么操作？",
#         "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
#         "item_names": ["HAK 180 烫金机"],
#         "history": mock_history,
#         "reranked_docs": mock_reranked_docs,
#         "is_stream": False,
#         "answer": None,
#     }
#     result = node_answer_output(mock_state)
#     print(result)
import sys
from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message
from utils.task_utils import add_running_task, add_done_task


@node_log("node_answer_output")
def node_answer_output(state:QueryGraphState):
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
    if not state["answer"]:
        state["answer"]="我是一个智能知识助手，下面是我找的相关图片"
        state["image_urls"]=[
            "http://192.168.64.2:9000/knowledge-base-files/upload-images/万用表RS-12的使用/16f1ae918c8905b9e4fc1d0c08fe8a9b55c34d69f4c1191f790bf08571419299.jpg",
            "http://192.168.64.2:9000/knowledge-base-files/upload-images/万用表RS-12的使用/17a896b47789994e2944e1590940a56fff9a93c68fed9b924cd572f4917cf087.jpg"
        ]
    save_chat_message(
        session_id= state.get("session_id"),
        role = "assistant",
        text = state.get("answer"),
        rewritten_query= state.get("rewritten_query"),
        item_names= state.get("item_names"),
        image_urls= state.get("image_urls")
    )
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
    return state