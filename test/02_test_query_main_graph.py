import json

from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from common.logging.logger import logger

logger.info("===== 开始测试 =====")

initial_state = create_query_default_state(session_id="test_001",
        original_query="HAK180烫金机怎么换电池")
every_step_state = None

# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息  stream_mode默认为updates
for event in query_app.stream(initial_state):
    for key, value in event.items():
        logger.success(f"节点: {key}")
        every_step_state = value

# 格式化输出最终状态
logger.info(f"最终状态: {json.dumps(every_step_state, indent=4, ensure_ascii=False)}")

logger.info("图结构:")
# uv add grandalf
query_app.get_graph().print_ascii()

logger.info("===== 测试结束 =====")