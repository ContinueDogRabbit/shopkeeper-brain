import sys
from pathlib import Path
import time
import shutil
import os

import requests

from common.logging.logger import logger, node_log, step_log
from common.config.mineru_config import mineru_config
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.task_utils import add_running_task, add_done_task
from utils.path_util import PROJECT_ROOT



@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取pdf_path 和 local_dir
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    # 判断是否为空
    if not pdf_path:
        # 如果pdf_path为空 抛出异常
        logger.error("状态中没有pdf_path,流程终止")
        raise ValueError("状态中没有pdf_path,流程终止")

    if not local_dir:
        # 如果local_dir为空 指定默认输出目录
        logger.warning("状态中没有local_dir,指定默认路径")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = local_dir

    # 转换为Path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 判断文件是否存在
    if not pdf_path_obj.is_file():
        # 如果pdf_path_path不存在或者不是文件  抛出异常
        logger.error(f"{pdf_path_obj}不存在或者不是文件")
        raise ValueError(f"{pdf_path_obj}不存在或者不是文件")


    # 如果local_dir_path不存在或者不是目录 创建新的目录
    if not local_dir_obj.is_dir():
        logger.warning(f"{local_dir_obj}不存在或者不是目录")
        # parents = True  同时创建父级目录    exist_ok=True  如果目录存在也不会报错
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj,local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj):
    # -------0. 校验MinerU配置---------
    if not mineru_config.api_key or not mineru_config.base_url:
        logger.error("MinerU配置信息有误，请检查！")
        raise ValueError("MinerU配置信息有误，请检查！")

    # -------1.申请上传地址---------
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": pdf_path_obj.name, "data_id": pdf_path_obj.stem}
        ],
        "model_version": "vlm"
    }

    # 发送请求
    response = requests.post(url, headers=header, json=data)
    # 获取请求响应的状态码
    http_status_code = response.status_code
    if http_status_code != 200:
        logger.error("向MinerU服务器发送请求获取文件上传地址失败")
        raise RuntimeError("向MinerU服务器发送请求获取文件上传地址失败")

    # 获取响应的内容   response.json()  以json形式获取数据(字典)   response.text 以字符串形式获取数据   response.content  以字节形式获取数据
    response_dict = response.json()
    # 获取业务接口返回的状态码
    response_dict_code = response_dict.get("code")
    # 获取业务接口返回的描述信息
    response_dict_msg = response_dict.get("msg")
    if response_dict_code != 0:
        logger.error(f"申请地址请求成功，但是业务处理失败，错误码：{response_dict_code},错误信息：{response_dict_msg}")
        raise RuntimeError(f"申请地址请求成功，但是业务处理失败，错误码：{response_dict_code},错误信息：{response_dict_msg}")

    batch_id = response_dict.get("data").get("batch_id")
    file_urls = response_dict.get("data").get("file_urls")

    if not batch_id:
        logger.error("没有返回batch_id")
        raise RuntimeError("没有返回batch_id")

    if not file_urls:
        logger.error("没有返回file_urls")
        raise RuntimeError("没有返回file_urls")

    file_upload_url = file_urls[0]

    logger.info(f">>>申请地址成功{file_upload_url}>>>")

    # -------2.上传文件---------
    # 向指定的位置上传文件
    # pdf_path_obj.read_text(en)   从文件中读取字符数据
    # pdf_path_obj.read_bytes()    从文件中读取字节数据
    # pdf_path_obj.write_text()    向文件写入字符数据
    # pdf_path_obj.write_bytes()   向文件写入字节数据

    file_upload_data = pdf_path_obj.read_bytes()
    # 注意：如果直接使用requests.put进行文件上传，有很大的概率会失败
    # 如果有代理，会在请求头中携带额外的信息，导致存储文件的服务器不能访问 而失败
    # requests.put(file_upload_url, data=file_upload_data)
    with requests.session() as session:
        # 纯净版的请求头,不随意携带代理的参数
        session.trust_env=False
        upload_response =  session.put(file_upload_url, data=file_upload_data)
        if upload_response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

    logger.info(f">>>向{file_upload_url}上传{pdf_path_obj}成功>>>")

    # -------3.轮询获取结果---------
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"

    # 超时时间
    timeout = 600000
    # 每隔多久轮询一次
    interval_time = 3
    # 轮询开始时间
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            logger.error("轮训获取解析结果超时")
            raise TimeoutError("轮训获取解析结果超时")

        try:
            poll_response = requests.get(poll_url, headers=header)
        except:
            logger.warning(f"请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue
        # 获取请求的状态码
        poll_http_status_code = poll_response.status_code
        # 判断是不是服务器内部的问题
        if poll_http_status_code != 200:
            if 500 <= poll_http_status_code < 600:
                logger.warning(f"可修复的网络异常,状态码为:{poll_http_status_code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常,状态码为:{poll_http_status_code}")
                raise RuntimeError(f"不可修复的网络状态异常,状态码为:{poll_http_status_code}")
        # 判断业务接口返回的内容
        poll_response_dict = poll_response.json()
        # 获取业务响应代码
        poll_response_dict_code = poll_response_dict.get("code")
        # 获取业务响应的消息
        poll_response_dict_msg = poll_response_dict.get("msg")
        if poll_response_dict_code != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict_code},失败信息:{poll_response_dict_msg}")
            raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict_code},失败信息:{poll_response_dict_msg}")

        extract_result = poll_response_dict.get("data").get("extract_result")[0]
        extract_result_state = extract_result.get("state")
        if extract_result_state == "done":
            extract_result_url = extract_result.get("full_zip_url")
            if not extract_result_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")
            logger.info(f">>>获取mineru服务器的解析结果：{extract_result_url}>>>")
            return extract_result_url
        elif extract_result_state == "failed":
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url, local_dir_path, stem):
    # -----从服务器下载文件----
    with requests.Session() as session:
        session.trust_env = False

        response = session.get(
            zip_url,
            timeout=(10, 120),
        )
        if response.status_code != 200:
            logger.error(f"从{zip_url}下载文件失败!")
            raise RuntimeError(f"从{zip_url}下载文件失败!")

    md_zip_path_obj:Path = local_dir_path / f"{stem}_result.zip"
    md_zip_path_obj.write_bytes(response.content)

    # -----解压到指定目录----
    extract_path_obj:Path = local_dir_path / stem
    if extract_path_obj.is_dir():
        # 如果解压的目标路径已经存在   先删除   避免旧数据对新数据造成污染
        shutil.rmtree(extract_path_obj)

    # 删除完后需要重新创建
    extract_path_obj.mkdir(parents=True, exist_ok=True)
    # 解压
    shutil.unpack_archive(md_zip_path_obj,extract_path_obj)

    # -----查找解压后的所有md文件 ->list ----
    md_file_list = list(extract_path_obj.rglob("*.md"))
    # 没有找到 MD 文件则抛出异常
    if not md_file_list:
        logger.error(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")
        raise RuntimeError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")

    # -----按优先级选择md文件----------
    # 优先级1: stem.md
    for md_file_obj in md_file_list:
        if md_file_obj.stem == stem:
            logger.info(f"文件{md_file_obj}解压成功")
            return md_file_obj
    # 优先级2: full.md
    target_md_obj = None
    for md_file_obj in md_file_list:
        if md_file_obj.name.lower() == "full.md":
            target_md_obj = md_file_obj
            logger.info(f"文件解压成功,但是名字是full.md,后续需要重命名")
            break
    # 优先级3: list[0]
    if not target_md_obj:
        target_md_obj = md_file_list[0]
    # 改名   stem.md
    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    logger.info(f"文件{md_file_obj}解压成功")
    return final_md_path_obj


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    """
    # TODO 1. 添加运行时列表
    add_running_task(state.get("task_id"),"node_pdf_to_md")
    # TODO 2. 从状态中获取数据并进行校验
    pdf_path_obj,local_dir_path =  step_1_validate_and_get_data(state)
    # TODO 3. 上传文件并解析
    zip_url = step_2_upload_and_poll(pdf_path_obj)
    # TODO 4. 下载并解压
    md_path = step_3_download_and_extract(zip_url,local_dir_path,pdf_path_obj.stem)
    state["md_path"] = md_path
    # TODO 5. 添加完成列表
    add_done_task(state.get("task_id"),"node_pdf_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)
    print(test_state)
    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")