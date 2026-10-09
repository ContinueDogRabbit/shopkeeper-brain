import json
import os
import re

from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task, add_done_task

# ====================== 全局配置（可根据模型调整）======================
# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于这个长度判定为短碎片，需要尝试合并
CHUNK_MIN = 400


@step_log("step_1_validate_get_data")
def step_1_validate_get_data(state):
    # 从状态中获取数据
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")

    # 校验是否存在
    if not md_content:
        if (not md_path) or (not Path(md_path).is_file()):
            # md_path为空,或者是不存在(不是文件)
            logger.error(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
            raise ValueError(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
        md_content = Path(md_path).read_text(encoding="utf-8")
        state["md_content"] = md_content

    if not file_title:
        # 文件名为空
        file_title = Path(md_path).stem
        logger.warning(f"状态中没有file_title为空,从md_path中获取:{file_title}")
        state['file_title'] = file_title

    # 注意：不同的操作系统，对于md文档中换行表现是不一样的    这里做一个统一   换行就用\n
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    return md_content, file_title, md_path


@step_log("step_2_split_document_by_title")
def step_2_split_document_by_title(md_content: str, file_title: str) -> list[dict[str, Any]]:
    # 最终返回的列表
    chunks: list[dict[str, Any]] = []

    # 当前激活的完整嵌套标题     例如：   # 标题1_## 标题2
    current_title: str | None = None

    # chunk正文内容缓存区     存放当前标题下的正文与代码行
    current_content_lines: list[str] = []

    # 孤儿数据缓存区
    orphan_lines: list[str] = []

    # 标记位  标记记录到孤儿数据缓存区的内容是否被成功合并到了首个chunk中
    has_flushed_first_chunk: bool = False

    # 标记位  标记是不是代码
    is_code: bool = False

    # 多级标题栈   : 用于标记H1~H6标题    例如:["H1","H2",None,"H4"]
    heading_stack: list[str] = []

    # 对md_content进行按行切割
    document_lines = md_content.split("\n")

    # 匹配标题的正则表达式
    title_reg = re.compile(r"^#{1,6}\s.+")

    # 对整个文档所有行进行遍历
    for line in document_lines:
        # 取出当前行的前后空格
        line_strip = line.strip()

        if not line_strip:
            # 如果是空行，直接跳过处理逻辑
            continue

        # 判断是不是代码标记
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            # 状态取反操作 ，对应的代码块的进入和跳出两种状态
            is_code = not is_code
            if not current_title:
                orphan_lines.append(line_strip)
            else:
                current_content_lines.append(line_strip)
            continue

        if is_code:
            # 如果是代码，将代码添加到孤儿数据列表或者内容列表中  保留代码格式
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)
            continue

        # 判断是不是标题
        if title_reg.match(line_strip):
            # 说明是标题
            # 获取当前标题级别
            heading_level = len(line_strip) - len(line_strip.lstrip("#"))
            if current_title and len(current_content_lines) > 0:
                if not has_flushed_first_chunk and len(orphan_lines) > 0:
                    # 如果有孤儿数据 ，将孤儿数据 放到正文前
                    current_content_lines = orphan_lines + current_content_lines
                    has_flushed_first_chunk = True
                    orphan_lines = []
                full_content = f"{current_title}\n" + "\n".join(current_content_lines)
                chunks.append({
                    "title": current_title,
                    "content": full_content,
                    "file_title": file_title
                })
                # 清空正文缓存区
                current_content_lines = []

            # 维护多级标题栈
            while len(heading_stack) < heading_level:
                # 处理跳级的情况
                heading_stack.append(None)

            # 保留当前栈中元素和标题层级匹配
            heading_stack = heading_stack[:heading_level]
            # 替换栈中对应的标题名
            heading_stack[heading_level - 1] = line_strip

            # 给current_title赋值  拼接heading_stack中不为空的标题
            current_title = "_".join([h for h in heading_stack if h])
            continue
        else:
            # 说明不是标题，当做正文进行处理
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)

    # 循环结束之后，对最后一个标题下的内容进行处理
    if current_title and (len(current_content_lines) > 0 or (not has_flushed_first_chunk and len(orphan_lines) > 0)):
        if not has_flushed_first_chunk and len(orphan_lines):
            current_content_lines = orphan_lines + current_content_lines
            has_flushed_first_chunk = True
            orphan_lines = []
        full_content = f"{current_title}\n" + "\n".join(current_content_lines)
        chunks.append({
            "title": current_title,
            "content": full_content,
            "file_title": file_title
        })
    elif (not current_title) and len(orphan_lines):
        chunks.append({
            "title": file_title,
            "content": "\n".join(orphan_lines),
            "file_title": file_title
        })
    logger.info(f"已经根据标题进行多级切块（极简稳健版），现有的块: {len(chunks)}")
    return chunks


@step_log("_split_chunk_content")
def _split_chunk_content(chunk) -> list[dict[str, Any]]:
    sub_chunks: list[dict[str, Any]] = []

    prefix = chunk.get("title") + "\n"
    content = chunk.get("content")
    deal_content = content[len(prefix):]

    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        # separators = ["。"]
    )

    for index, text in enumerate(spliter.split_text(deal_content), start=1):
        sub_chunks.append({
            "file_title": chunk.get("file_title"),
            "parent_title": chunk.get("title"),
            "title": f"{chunk.get('title')}_{index}",
            "part": index,
            "content": prefix + text
        })
    return sub_chunks


@step_log("_merge_chunk_content")
def _merge_chunk_content(refine_chunks) -> list[dict[str, Any]]:
    merge_refine_chunks: list[dict[str, Any]] = []

    # 核心思路：指针
    # 每次循环用基准chunk和next_chunk进行合并(满足条件)    base_chunk = next_chunk
    # 定义一个基准chunk，用于和下一个chunk进行比较
    base_chunk: dict[str, Any] = None

    # 对精细化切分后的所有chunk进行遍历
    for next_chunk in refine_chunks:
        if base_chunk is None:
            base_chunk = next_chunk
            continue
        # 判断长度是否小于CHUNK_MIN
        is_too_long = len(base_chunk.get("content")) > CHUNK_MIN
        if not is_too_long:
            # base_chunk的内容<CHUNK_MIN
            # 判断是不是在同一个标题下
            is_same_parent_title = (base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get(
                "parent_title"))
            if is_same_parent_title:
                # 在同一个标题下
                # 判断base_chunk + next_chunk 内容加起来之后是否大于1000
                base_content: str = base_chunk.get("content")
                next_cleared_content: str = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
                is_merged_too_long = (len(base_content) + len(next_cleared_content)) > CHUNK_MAX_SIZE
                if not is_merged_too_long:
                    # base_chunk和next_chunk合并后不大于1000
                    base_chunk['content'] = base_content + "\n" + next_cleared_content
                else:
                    merge_refine_chunks.append(base_chunk)
                    base_chunk = next_chunk
            else:
                # base_chunk和next_chunk不在同一个标题下
                merge_refine_chunks.append(base_chunk)
                base_chunk = next_chunk
        else:
            # base_chunk的内容 > CHUNK_MIN
            merge_refine_chunks.append(base_chunk)
            base_chunk = next_chunk

    if base_chunk:
        merge_refine_chunks.append(base_chunk)
    logger.info(f"完成小于400的chunk的合并,合并后的数量:{len(merge_refine_chunks)}")
    return merge_refine_chunks


@step_log("step_3_refine_split_and_merge_chunks")
def step_3_refine_split_and_merge_chunks(title_chunks) -> list[dict[str, Any]]:
    refine_chunks: list[dict[str, Any]] = []
    # 遍历语义切分后的所有chunk
    for chunk in title_chunks:
        # 判断当前chunk的内容长度是不是大于了CHUNK_SIZE
        if len(chunk.get("content")) > CHUNK_SIZE:
            refine_chunks.extend(_split_chunk_content(chunk))
        else:
            refine_chunks.append(chunk)
    logger.info(f"chunks经过超长以后向短切割处理! 切割后的数量:{len(refine_chunks)}")
    # for chunk in refine_chunks:
    #     logger.success(chunk)

    # 如果经过精细化切分之后，有些chunk的内容长度 < CHUNK_MIN400，那么对其进行合并，但是要求合并后大小不能超过CHUNK_MAX_SIZE 1000
    refine_chunks = _merge_chunk_content(refine_chunks)
    # for chunk in refine_chunks:
    #     logger.error(chunk)
    return refine_chunks


@step_log("step_4_padding_chunks_metadata")
def step_4_padding_chunks_metadata(define_chunks):
    # 补充parent_title和part属性信息
    for chunk in define_chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk.get("title")

        if "part" not in chunk:
            chunk["part"] = 1
    logger.info(f"完成chunks的元素补充,所有属性都完整!")


@step_log("step_5_backup_chunks_json")
def step_5_backup_chunks_json(define_chunks, md_path):
    # 定义要写到磁盘的路径
    md_path_obj = Path(md_path)
    json_path_obj: Path = md_path_obj.parent / f"{Path(md_path).stem}.json"

    # 将chunk内容写到磁盘中   json.dumps将python对象转换为json字符串
    json_path_obj.write_text(json.dumps(define_chunks, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info(f"完成chunks数据的备份,备份位置:{str(json_path_obj)}")


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    """
    # TODO 1. 添加到运行时列表
    add_running_task(state.get("task_id"), "node_document_split")
    # TODO 2. 从状态中获取数据并进行校验
    md_content, file_title, md_path = step_1_validate_get_data(state)
    # TODO 3. 根据标题进行语义切割   list[{"title":"","content":"","file_title":""}]   title:md文档的标题    file_title:文件名
    title_chunks: list[dict[str, Any]] = step_2_split_document_by_title(md_content, file_title)

    # TODO 4. 精细化切割
    define_chunks: list[dict[str, Any]] = step_3_refine_split_and_merge_chunks(title_chunks)

    # TODO 5. 补充元数据信息 (parent_title,part)
    step_4_padding_chunks_metadata(define_chunks)

    # for chunk in define_chunks:
    #     logger.success(chunk)

    # TODO 6. 将切割后的chunk持久化到磁盘上
    step_5_backup_chunks_json(define_chunks, md_path)
    # TODO 7. 更新状态
    state["chunks"] = define_chunks
    # TODO 8. 添加到已完成列表
    add_done_task(state.get("task_id"), "node_document_split")
    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT
    from processor.import_processor.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册_new.md")
    # test_md_name = os.path.join(r"output\hak180产品安全手册", "test.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        # result_state = node_md_img(test_state)
        # logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        # logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        # final_chunks = final_state.get("chunks", [])
        # logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")