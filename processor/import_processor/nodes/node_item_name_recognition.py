import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.path_util import PROJECT_ROOT
from utils.task_utils import add_running_task, add_done_task

# 主体识别上下文切片数：取前 K 个切片用于 LLM 识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 主体识别上下文总字符数上限：防止上下文过长导致大模型输入超限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取chunks
    chunks = state.get("chunks")
    # 从状态中获取file_title
    file_title = state.get("file_title")

    # 从状态中或者md_path
    md_path = state.get("md_path")
    # 如果从状态中没有找到chunks,尝试从本地读取
    md_path_obj = Path(md_path)
    if not chunks:
        if md_path_obj.is_file():
            # json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
            # 只是在内存层面上创建新的Path对象，对磁盘文件没有影响
            json_path_obj:Path = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                # 从json文件中读取chunks信息
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                # 将chunks信息更新到状态中
                state["chunks"] = chunks
                logger.warning("从备份文件中读取chunks")
            else:
                logger.error(f"chunks没有值,同时也没有读取到对应json_path,抛出异常!")
                raise ValueError(f"chunks没有值,同时也没有读取到对应json_path,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")

    if not file_title:
        file_title = md_path_obj.stem
        state["file_title"] = file_title
        logger.warning(f"file_title不存在,给与默认值:{file_title}")
    return chunks, file_title

@step_log("step_2_call_llm_return_item_name")
def step_2_call_llm_return_item_name(chunks, file_title):
    """
        调用大语言模型，提取商品名称
        chunk字典:
            里面原有属性:{"parent_title":"","content":"","part":"","title":"","file_title":""}
            需要的属性:{"parent_title":"","content":""}
            拼接提示词的时候: 标题：parent_title，内容：content\n
    """
    # 获取模型对象
    llm = get_llm_client()
    # 拼接提示词
    context = ""
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        context += f"标题：{chunk.get("parent_title")}，内容：{chunk.get("content")}\n"
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    # 加载提示词模版
    prompt_text = load_prompt(name="item_name_recognition",file_title=file_title,context=context)
    message = HumanMessage(
        content=prompt_text
    )
    # 拼接调用链
    chains = llm | StrOutputParser()

    # 调用模型   获取提取到的商品名称
    item_name = chains.invoke([message])

    # 判断商品名是否提取成功   如果没有提取成功用文件名file_title替代
    if not item_name:
        item_name = file_title
        logger.warning(f"没有识别出item_name,使用file_title赋值:{item_name}")
    # 返回结果
    return item_name

@step_log("step_3_padding_item_name_to_chunks")
def step_3_padding_item_name_to_chunks(chunks, item_name):
    # 向chunk中填入item_name
    for chunk in chunks:
        chunk["item_name"] = item_name

@step_log("step_4_prepared_item_name_collection")
def step_4_prepared_item_name_collection():
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 判断集合是否存在
    has_collection = milvus_client.has_collection(collection_name=milvus_config.item_name_collection)
    # 如果存在，不需要创建，直接返回
    if has_collection:
        logger.info(f"{milvus_config.item_name_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.item_name_collection}不存在,进行集合的创建!")
    # 创建schema
    schema = milvus_client.create_schema(
        auto_id=True,  # 主键是否自增
        enable_dynamic_field=True,
    )
    # 向schema中添加字段
    schema.add_field(field_name="pk",datatype=DataType.INT64,is_primary=True)
    schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535)
    schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535)
    schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
    schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 创建索引 设置索引参数
    index_params = milvus_client.prepare_index_params()
    # 给稠密向量添加索引
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,  # 相邻最大的节点数量
            "efConstruction": 100  # 候选的节点数量
        }
    )
    # 给稀疏向量添加索引
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )
    # 创建集合
    milvus_client.create_collection(
        collection_name=milvus_config.item_name_collection,
        schema=schema,
        index_params=index_params
    )

    logger.info(f"{milvus_config.item_name_collection}创建成功~")


@step_log("step_5_insert_item_name_data")
def step_5_insert_item_name_data(item_name, file_title):
    # 调用嵌入模型给item_name生成向量  注意：generate_embeddings接收的参数是文档列表
    #    result->{"dense":[[1024...],[1024...],[1024...]...],"sparse":[{k->tokenid,v->分值},{},{}....]}
    result = generate_embeddings([item_name])
    dense_vector = result["dense"][0]
    sparse_vector = result["sparse"][0]

    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 先删除对应的数据   filter过滤相当于在mysql数据中执行删除where后的过滤条件  delete from 表  where
    milvus_client.delete(
        collection_name= milvus_config.item_name_collection,
        filter=f"file_title == '{file_title}'"   # 在milvus中判断是否相等用==
    )
    # 向集合中插入数据
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data={
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector":sparse_vector
        }
    )
    logger.info(f"向集合{milvus_config.item_name_collection}中插入数据成功")


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    """
    # TODO 1. 添加运行时状态
    add_running_task(task_id=state.get("task_id"),node_name="node_item_name_recognition")
    # TODO 2. 从状态中获取数据并进行校验
    chunks, file_title = step_1_validate_and_get_data(state)
    # TODO 3. 调用模型提取item_name
    item_name: str = step_2_call_llm_return_item_name(chunks, file_title)
    # TODO 4. 回填数据   --- chunk中添加item_name
    step_3_padding_item_name_to_chunks(chunks, item_name)
    # TODO 5. 创建Milvus集合(包含schema结构以及索引)
    step_4_prepared_item_name_collection()
    # TODO 6. 向量化 并向Milvus中插入数据
    step_5_insert_item_name_data(item_name, file_title)

    state["item_name"] = item_name
    state["chunks"] =  chunks
    # TODO 7.添加结束状态
    add_done_task(task_id=state.get("task_id"),node_name="node_item_name_recognition")
    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            "md_path":PROJECT_ROOT/"output/hak180产品安全手册/hak180产品安全手册_new.md",
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "parent_title":"产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "parent_title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "parent_title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # # 4. 验证Milvus存储（可选）
        # milvus_client = get_milvus_client()
        # collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        # if milvus_client and collection_name:
        #     milvus_client.load_collection(collection_name)
        #     # 检索测试结果
        #     item_name = result_state.get('item_name')
        #     safe_name = escape_milvus_string(item_name)
        #     res = milvus_client.query(
        #         collection_name=collection_name,
        #         filter=f'item_name=="{safe_name}"',
        #         output_fields=["file_title", "item_name"]
        #     )
        #     logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()