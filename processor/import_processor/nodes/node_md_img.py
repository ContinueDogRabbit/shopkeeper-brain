import sys
import os
import re
import base64
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from common.config.minio_config import minio_config
from common.logging.logger import logger, node_log, step_log
from common.config.lm_config import lm_config
from processor.import_processor.state import ImportGraphState
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state)->tuple[str, Path,Path]:
    # 从状态中获取md_path
    md_path = state.get("md_path")
    # 判断状态中是否存在md_path
    if not md_path:
        logger.error("在状态中不存在md_path属性")
        raise RuntimeError("在状态中不存在md_path属性")
    # 封装Path对象
    md_path_obj = Path(md_path)

    # 判断md文件是否存在
    if not md_path_obj.exists():
        logger.error(f"{md_path_obj}不存在")
        raise FileNotFoundError(f"{md_path_obj}不存在")

    # 获取文件内容
    md_content = md_path_obj.read_text(encoding="utf-8")

    # 将md_content放到状态中
    state["md_content"] = md_content

    # 获取存放图片的路径
    md_images_dir_obj = md_path_obj.parent / "images"

    return md_content, md_path_obj,md_images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content, md_images_dir_obj)->list[tuple[str,str,tuple[str,str]]]:
    """
    获取图片信息以及上下文
    :param md_content:   md文件内容
    :param md_images_dir_obj:   图片所在路径
    :return: list[(图片名,图片路径,(上文,下文))]
    """
    image_info_list = []
    # 遍历md_images_dir_obj目录  获取目录中的每一个图片
    for image_file_obj in md_images_dir_obj.iterdir():
        # 获取图片名
        image_name = image_file_obj.name
        # 获取图片地址
        image_path = str(image_file_obj)
        # 过滤掉不支持的图片类型
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}是不支持的图片类型")
            continue
        # 定义正则表达式
        pattern = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+ r".*?\)")
        # 根据正则匹配md所有的图片
        search_match = pattern.search(md_content)
        # 判断是否匹配上
        if not search_match:
            logger.warning(f"{image_name}没有被md_content引用引用,跳过,直接下一次!!")
            continue
        # 获取匹配内容的起始下标
        start = search_match.start()
        # 获取匹配内容的结束下标
        end = search_match.end()
        # 如果匹配上  获取上下文
        pre_context = md_content[max(0,start-100):start]
        post_context = md_content[end:min(end + 100,len(md_content))]
        logger.debug( f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文:{pre_context} , 下文:{post_context}")
        image_info_list.append((image_name,image_path,(pre_context,post_context)))
    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    return image_info_list


@step_log("step_3_image_summary")
def step_3_image_summary(image_info_list, root_folder)->dict[str, str]:
    """
    调用vm模型，给图片生成摘要
    :param image_info_list:   图片信息列表
    :param root_folder:       md文件名---给提示词文件中的变量用
    :return:   字典dict[图片名:摘要]
    """
    image_summary_dict = {}
    # 获取模型对象
    vl_model = get_llm_client(model=lm_config.vl_model)
    # 循环image_info_list数据,获取每一张图片的信息
    for image_name,image_path,image_content in image_info_list:

        # 封装提示词
        image_prompt_text = load_prompt("image_summary",root_folder = root_folder,image_content = image_content)

        # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        # image_data = base64.b64encode(requests.get(image_url).content).decode("utf-8")
        image_path_obj = Path(image_path)
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": image_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"},
                },
            ]
        )
        # 调用视觉模型获取结果
        chains = vl_model | StrOutputParser()
        image_summary = chains.invoke([message])
        # 添加范围内限制
        apply_api_rate_limit()
        # 拼接提示返回结果到字典中
        image_summary_dict[image_name] = image_summary
        logger.debug(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")
    return image_summary_dict


@step_log("step_4_upload_images_get_url")
def step_4_upload_images_get_url(image_info_list, stem)->dict[str,str]:
    # 获取操作minio的客户端对象
    minio_client = get_minio_client()
    # 获取对应目录下的所有图片  --- list_objects
    image_list = minio_client.list_objects(
        bucket_name = minio_config.bucket_name,
        # 注意：不能以/开头，否则查询不到数据
        prefix= minio_config.minio_img_dir[1:] + "/" + stem,
        recursive=True
    )
    # 先删除掉所有图片         --- remove_objects
    # 封装要要删除的对象为DeleteObject类型
    delete_object_list = [DeleteObject(obj.object_name) for obj in image_list]
    errors = minio_client.remove_objects(
        bucket_name = minio_config.bucket_name,
        delete_object_list= delete_object_list
    )

    # remove_objects 底层是生成器    是惰性执行的，需要通过for循环进行触发删除操作的执行
    for error in errors:
        logger.warning(f"删除图片出现问题:{error}")

    # 重新上传图片            --- fput_object
    # 对本地图片列表进行遍历
    image_url_dict = {}
    for image_name,image_path,_ in image_info_list:
        try:
            minio_client.fput_object(
                bucket_name = minio_config.bucket_name,
                # /upload-images/hak180烫金机操作手册/xxx.jpg
                object_name = minio_config.minio_img_dir + "/" + stem + "/" + image_name,
                file_path = image_path,
                content_type=guess_type(image_name)[0]
            )
            url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
            image_url_dict[image_name] = url
            logger.debug(f"{image_name}已经完成上传,对应的地址为:{url}")
        except:
            logger.warning(f"{image_name}上传失败,跳过,继续下一张图片传递!!")
    return image_url_dict

@step_log("step_5_md_content_image_replace")
def step_5_md_content_image_replace(md_content, image_summary_dict, image_url_dict):
    # 对摘要字典进行遍历
    for image_name,image_summary in image_summary_dict.items():
        image_url = image_url_dict.get(image_name)

        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name) + r".*?\)")

        # 使用正则替换md文件中的内容
        # 注意：在使用sub进行替换的操作的时候，如果替换的内容中有 / \ 这样的内容，可能会被re识别为group的获取
        # 解决：使用匿名函数(lambda)   匿名函数的作用：将/\特殊字符也当做普通字符进行处理
        # reg.sub(f"![{summary}]({image_url})",md_content)
        md_content = reg.sub(lambda _: f"![{image_summary}]({image_url})",md_content)
        logger.debug(f"已经完成:{image_name}图片的替换,替换入的描述:{image_summary},替换的地址:{image_url}")
    return md_content


@step_log("step_6_backup_new_md_content")
def step_6_backup_new_md_content(md_content_new, md_path_obj):
    # 创建一个新的路径对象
    md_path_obj_new: Path = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")

    # 将新的md内容写到新的对象中
    md_path_obj_new.write_text(data=md_content_new, encoding="utf-8")

    logger.info(f"已经将新的md_content内容备份到:{str(md_path_obj_new)}")
    # 3. 返回新的Path对象
    return md_path_obj_new


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    """
    # TODO 1. 添加运行时状态
    add_running_task(state.get("task_id"),"node_md_img")
    # TODO 2. 从状态中获取数据并进行校验  返回md_content md_path_obj md_images_dir_obj
    md_content, md_path_obj ,md_images_dir_obj = step_1_validate_and_get_data(state)
    # 判断是否有图片，如果md中没有图片，后续的所有操作，都不需要做了
    if (not md_images_dir_obj.is_dir()) or len(list(md_images_dir_obj.iterdir()))==0:
        logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
        return state
    # TODO 3. 获取图片信息以及上下文   list[tuple[图片名,图片地址,tuple[上文,下文]]]
    image_info_list: list[tuple[str,str,tuple[str,str]]] =  step_2_scan_images(md_content, md_images_dir_obj)
    # TODO 4. 调用视觉模型  生成图片的摘要
    # list[tuple[str,str,tuple[str,str]]]->dict{k:图片名,v:摘要}
    image_summary_dict: dict[str, str] = step_3_image_summary(image_info_list, md_path_obj.stem)

    # TODO 5. 将图片上传到minio服务器
    image_url_dict: dict[str, str] = step_4_upload_images_get_url(image_info_list, md_path_obj.stem)

    # TODO 6. 替换md_content中图片内容
    md_content_new: str = step_5_md_content_image_replace(md_content, image_summary_dict, image_url_dict)

    # TODO 7. 将新的md文件进行磁盘存储
    md_path_obj_new: Path = step_6_backup_new_md_content(md_content_new, md_path_obj)

    # 更新状态
    state["md_content"] = md_content_new
    state["md_path"] = md_path_obj_new

    # TODO 8. 添加完成状态
    add_done_task(state.get("task_id"),"node_md_img")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册.md")
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")