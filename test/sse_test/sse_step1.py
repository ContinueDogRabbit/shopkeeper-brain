import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义一个 GET 接口，路径：/simple_stream
# 这是一个 SSE 服务端推送接口
@app.get("/simple_stream")
async def simple_stream():
    # 定义一个 异步生成器函数，用来源源不断产生消息
    # 这是核心：yield 一条一条推送数据，不会一次性返回
    async def event_generator():
        # 循环 5 次，推送 5 条消息
        for i in range(5):
            # SSE 协议标准格式：
            # data: 消息内容\n\n
            # 必须以 data: 开头，必须用 \n\n 表示本条消息结束
            yield f"data: 这是第{i + 1}条测试消息\n\n"
            # 异步等待 1 秒，模拟每隔1秒推送一条消息
            await asyncio.sleep(1)

    # 返回 StreamingResponse，实现流式推送
    # media_type="text/event-stream" 告诉浏览器这是 SSE 服务端推送
    return StreamingResponse(
        event_generator(),  # 传入异步生成器，源源不断产出数据
        media_type="text/event-stream"  # 声明 SSE 格式
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)