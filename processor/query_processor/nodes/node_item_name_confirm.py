import time
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from common.config.milvus_config import milvus_config
from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task



# 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
ITEM_NAME_CONFIRM_THRESHOLD = 0.70
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.60
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2
ITEM_NAME_CONFIRMED_TOPK = 1


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取数据
    session_id = state.get("session_id")
    original_query = state.get("original_query")

    # 非空判断
    if (not session_id) or (not original_query):
        logger.error("session_id或者original_query是空，不能继续后续业务~")
        raise ValueError("session_id或者original_query是空，不能继续后续业务~")
    return original_query, session_id


@step_log("step_2_get_history_by_session_id")
def step_2_get_history_by_session_id(session_id):
    # 根据session_id到mongodb中获取当前会话对应的聊天记录
    history_list:list[dict[str,Any]] =  get_recent_messages(session_id)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    # 过滤出有效的历史记录     有效:记录中的item_name不能为空
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    return history_list


@step_log("step_3_call_llm_rewritten_and_extract_itemnames")
def step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query):
    """
        {
        _id:ObjectId(1),
        role: user / assistant
        text: user-> original_query  assistant -> answer
        rewritten_query => 重写的问题
        item_names = [] 关联的item_name 不为空!!
        image_urls = []
        ts = 时间戳...
       }
    """
    # 对历史记录列表里的数据进行拼接处理---->得到字符串
    # 不同类型的角色，拼接的内容也是不一样的
    # 判断是否有历史记录
    if history_list:
        # 在mongodb中有当前会话对应的历史记录
        history_text_list :list[dict[str,Any]] = []
        # 对历史记录进行遍历
        for index,history_item in enumerate(history_list,start=1):
            # 判断当前取出的这条历史记录是用户角色还是助手角色
            if history_item.get("role") == "user":
                # 用户角色  编号:xxx,用户提问,原始问题:xx,重写后的问题:xxx，关联item_name：xxx
                history_text_list.append(f"编号:{index},用户提问,原始问题:{history_item.get('text')},"
                                         f"重写后的问题:{history_item.get('rewritten_query')}，"
                                         f"关联item_name：{','.join(history_item.get('item_names'))}")
            else:
                # 助手角色  编号:xxx,助手回答,重写问题:xx,回答结果:xx,关联item_name:xxx"
                # 注意：助手回答内容可能比较多，为了避免内容过多可以使用如下方式进行优化
                # 方式1：对历史记录中的回答内容进行截取
                # 方式2： 调整提示词模版的内容
                history_text_list.append(f"编号:{index},助手回答,重写问题:{history_item.get('rewritten_query')},"
                                         f"回答结果:{history_item.get('text')[:50]},关联item_name:{','.join(history_item.get('item_names'))}")
        history_text = "\n".join(history_text_list)
    else:
        # 在mongodb中没有当前会话对应的历史记录
        history_text = "没有有效的历史聊天记录!"

    # 获取大语言模型
    llm_model = get_llm_client()
    # 加载提示词模版 生成提示词
    prompt_text = load_prompt("rewritten_query_and_itemnames",query = original_query,history_text=history_text)
    # 封装消息对象
    message = HumanMessage(
        content=prompt_text
    )
    # 封装调用链
    # 面试题: 如果希望模型返回json格式数据，应该如何处理?
    # 1. 提示词中指定返回json格式
    # 2. 在提示词中提供样例参考
    # 3. 在模型中加参数  model_kwargs["response_format"] = {"type": "json_object"}
    # 4. 使用json输出格式化解析器  JsonOutputParser()

    chains = llm_model | JsonOutputParser()
    # 执行以及获取结果  {"item_names":[],"rewritten_query":""}
    json_dict = chains.invoke([message])
    # 对结果进行处理   item_names没提取出来 直接赋值[],如果rewritten_query没重写，直接将原始问题赋值给rewritten_query
    if "item_names" not in json_dict:
        json_dict["item_names"] = []

    if "rewritten_query" not in json_dict:
        json_dict["rewritten_query"] = original_query

    # 返回结果
    return json_dict

@step_log("step_4_select_item_names_milvus")
def step_4_select_item_names_milvus(llm_item_names)->dict[str, list[dict]]:
    # 定义一个字典用于接收返回的结果
    milvus_search_dict:dict[str, list[dict]] = {}

    # 对模型生成的主体名进行向量化处理
    # llm_item_names [商品A,商品B]
    # result: {
    # "dense":[[商品A向量化后的稠密数据1024],[商品B向量化后的稠密数据1024]],
    # "sparse":[{商品A向量化后的稀疏数据},{商品B向量化后的稀疏数据}]
    # }
    result = generate_embeddings(llm_item_names)
    # 对模型提取所有主体名进行遍历
    for index,llm_item_name in enumerate(llm_item_names):
        # 获取当前遍历出来的主体名对应的稠密向量
        dense_vector = result.get("dense")[index]
        # 获取当前遍历出来的主体名对应的稀疏向量
        sparse_vector = result.get("sparse")[index]

        # 封装AnnSearchRequest
        reqs_list = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )

        # 使用混合检索对稀疏和稠密的结果进行汇合
        milvus_client = get_milvus_client()
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs_list,
            ranker_weights=(0.4, 0.6),
            norm_score=True, #是否进行归一化处理
            output_fields=["item_name"]
        )

        # 对混合检索返回的结果数据进行处理
        #response       [[{id:xxx,distance:xx,entity{"item_name":xxx}},{id:xxx,distance:xx,entity{"item_name":xxx}},{},{},{}]]
        # real_response [{id:xxx,distance:xx,entity{"item_name":xxx}},{id:xxx,distance:xx,entity{"item_name":xxx}},{},{},{}]
        real_response = response[0]

        llm_item_result_list: list[dict] = []
        # 遍历当前主体名混合检索查询结果
        for item in real_response:
            llm_item_result_list.append({
                "item_name":item.get("entity").get("item_name"),
                "score": item.get("distance")
            })
        # {"商品A":[{"item_name":"商品a","score":0.7},{{"item_name":"商品ab","score":0.6}}]}
        milvus_search_dict[llm_item_name] = llm_item_result_list
    return milvus_search_dict


@step_log("step_5_select_confirmed_and_option_item_names")
def step_5_select_confirmed_and_option_item_names(milvus_dict):
    """

    :param milvus_dict:
    {
    "商品A":[{"item_name":"商品a","score":0.7},{"item_name":"商品ab","score":0.6}]
    "商品B":[{"item_name":"商品b","score":0.7},{"item_name":"商品ab","score":0.6}]
    }
    :return:  {"confirmed_list":[item_name1],"option_list":[item_name2,...]}"}
    """
    confirmed_list: list[str] = []
    option_list: list[str] = []
    for llm_item_name,item_list in milvus_dict.items():
        high_list = [item.get("item_name") for item in item_list if item.get("score")>=ITEM_NAME_CONFIRM_THRESHOLD]
        middle_list = [item.get("item_name") for item in item_list if
                       ITEM_NAME_CANDIDATE_THRESHOLD <= item.get("score", 0.0) < ITEM_NAME_CONFIRM_THRESHOLD]
        if high_list:
            # 不为空
            # 高分 -> 1
            confirmed_list.extend(
                high_list[:ITEM_NAME_CONFIRMED_TOPK]
            )
            logger.info(f"{llm_item_name}已经获取到可信的主体:{high_list[:ITEM_NAME_CONFIRMED_TOPK]}")
            continue
        if middle_list:
            option_list.extend(
                middle_list[:ITEM_NAME_OPTIONS_TOPK]
            )
            logger.info(f"{llm_item_name}没有可信的主体,但是有可选的主体:{middle_list[:ITEM_NAME_OPTIONS_TOPK]}")
    return {
        "confirmed": confirmed_list,  # expr item_name in [名字,名字]
        "option": option_list
    }


@step_log("step_6_change_state_property")
def step_6_change_state_property(state, rewritten_query, confirmed_option_dict):
    # 获取确认列表
    confirmed_list = confirmed_option_dict.get("confirmed")
    # 获取可选列表
    option_list = confirmed_option_dict.get("option")

    # 判断确认列表中是否有数据
    if confirmed_list:
        state["rewritten_query"] = rewritten_query
        state["item_names"] = confirmed_list
        logger.info(f"已经有明确的item_name列表,已经更新state item_name以及rewritten_query属性!")
        return
    if option_list:
        state['answer'] = f"没有可信的item_name,但是有可选的:{option_list},请您明确!"
        logger.info(f"没有明确的item_name列表,但是有可选的!已经更新answer属性!")
        return

    # 如果没有确认 也没有可选
    state['answer'] = f"没有可信,也没有可选的实体列表,请明确以后再次问题!!"


@step_log("step_7_change_state_property")
def step_7_save_user_chat_message(state):
    # 向mongodb中写入数据
    save_chat_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names")
    )


@node_log("node_item_name_confirm")
def node_item_name_confirm(state:QueryGraphState):
    """
    节点功能：确认用户问题中的核心商品名称。
    """
    # TODO 1.添加到运行时任务列表
    add_running_task(state.get("session_id"),sys._getframe().f_code.co_name,is_stream=state.get("is_stream"))
    # TODO 2.从状态中获取数据并进行校验
    original_query, session_id = step_1_validate_and_get_data(state)
    # TODO 3.根据session_id获取历史记录
    history_list: list[dict] = step_2_get_history_by_session_id(session_id)
    # TODO 4.调用模型提取主体重写问题  {"item_names":[],"rewritten_query":""}
    llm_dict: dict[str, Any] = step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query)
    confirmed_option_dict = {}
    if llm_dict.get("item_names"):
        # TODO 5.对模型生成的主体名进行向量化查询  {"商品A":[{"item_name":"商品a","score":0.7},{{"item_name":"商品ab","score":0.6}}]}
        milvus_dict: dict[str, list[dict]] = step_4_select_item_names_milvus(llm_dict.get("item_names"))
        # TODO 6  .封装确认和可选列表
        confirmed_option_dict: dict[str, list[str]] = step_5_select_confirmed_and_option_item_names(milvus_dict)
    # TODO 7.根据确认和可选更新状态
    step_6_change_state_property(state, llm_dict.get("rewritten_query"), confirmed_option_dict)
    # TODO 8.向mongodb中写入对话
    step_7_save_user_chat_message(state)
    # TODO 9.添加到已完成时任务列表
    add_done_task(state.get("session_id"),sys._getframe().f_code.co_name,is_stream=state.get("is_stream"))
    return state

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 190 烫金机怎么用？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)
# import sys
# from common.logging.logger import node_log
# from processor.query_processor.state import QueryGraphState
# from utils.clients.mongo_history_utils import save_chat_message
# from utils.task_utils import add_running_task, add_done_task
#
#
# @node_log("node_item_name_confirm")
# def node_item_name_confirm(state:QueryGraphState):
#     add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
#     save_chat_message(
#         session_id=state.get("session_id"),
#         role="user",
#         text=state.get("original_query"),
#         rewritten_query=state.get("rewritten_query"),
#         item_names=state.get("item_names")
#     )
#     add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, is_stream=state.get("is_stream"))
#     return state
