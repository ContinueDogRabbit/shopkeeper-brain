from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.node_answer_output import node_answer_output
from processor.query_processor.nodes.node_item_name_confirm import node_item_name_confirm
from processor.query_processor.nodes.node_rerank import node_rerank
from processor.query_processor.nodes.node_rrf import node_rrf
from processor.query_processor.nodes.node_search_embedding import node_search_embedding
from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from processor.query_processor.nodes.node_web_search_mcp import node_web_search_mcp
from processor.query_processor.state import QueryGraphState

# 创建主图对象
builder = StateGraph(QueryGraphState)

# 添加节点
builder.add_node("node_item_name_confirm",node_item_name_confirm)
builder.add_node("node_search_embedding",node_search_embedding)
builder.add_node("node_search_embedding_hyde",node_search_embedding_hyde)
builder.add_node("node_web_search_mcp",node_web_search_mcp)
builder.add_node("node_rrf",node_rrf)
builder.add_node("node_rerank",node_rerank)
builder.add_node("node_answer_output",node_answer_output)
# ---添加边---
# 设置起始节点
# builder.add_edge(START,"node_item_name_confirm")
builder.set_entry_point("node_item_name_confirm")
# 定义路由函数
def after_node_item_name_confirm(state: QueryGraphState):
    if state.get("answer"):
        """
        这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
        具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
        1. 多选一（反问用户） ：
        - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，且置信度都不足以直接确认。
        - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
        - 结果 ：此时不需要再去检索文档了，直接把这句话发给用户让他选。
        2. 查无此人（拒绝回答） ：

        - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，但库里只有华为的数据），或者评分过低（<0.6）。
        - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
        - 结果 ：同样不需要后续检索，直接结束流程。
        """
        return "node_answer_output"

    return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"

# 添加路由边
builder.add_conditional_edges(
    "node_item_name_confirm",
    after_node_item_name_confirm,
    {
        "node_answer_output":"node_answer_output",
        "node_search_embedding":"node_search_embedding",
        "node_search_embedding_hyde":"node_search_embedding_hyde",
        "node_web_search_mcp":"node_web_search_mcp"
    }
)
# 添加普通边
builder.add_edge("node_search_embedding","node_rrf")
builder.add_edge("node_search_embedding_hyde","node_rrf")
builder.add_edge("node_web_search_mcp","node_rrf")
builder.add_edge("node_rrf","node_rerank")
builder.add_edge("node_rerank","node_answer_output")
builder.add_edge("node_answer_output",END)

# 编译
query_app = builder.compile()

