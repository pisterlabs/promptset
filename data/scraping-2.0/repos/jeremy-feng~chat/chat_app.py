from config import Config
from server import app

from dash.dependencies import Input, Output, State
import feffery_markdown_components as fmc
import feffery_utils_components as fuc
import feffery_antd_components as fac
from datetime import datetime
from dash import html, dcc
import openai
import dash
import os

# 本地运行需要开启代理
import sys

if len(sys.argv) > 1 and sys.argv[1] == "local":
    import os

    os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"

# 载入 openai api key
openai.api_key = Config.openai_api_key
feature = """
在上方，你可以：

1. 开启多轮对话模式，我将记住你之前的问题。

2. 导出当前对话记录为 Markdown 文件，你可以将其保存到本地。

3. 一键清空当前对话记录。
"""
server = app.server
app.layout = fac.AntdWatermark(
    [
        # 注入问题返回状态消息提示
        html.Div(id="response-status-message"),
        # 注入历史对话记录存储
        dcc.Store(id="multi-round-store", data={"status": "关闭", "history": []}),
        # 注入问答记录 markdown 下载
        dcc.Download(id="history-qa-records-download"),
        html.Div(
            fuc.FefferyDiv(
                [
                    fac.AntdRow(
                        [
                            fac.AntdCol(
                                fac.AntdParagraph(
                                    [
                                        fac.AntdText(
                                            "在线问答机器人",
                                            strong=True,
                                            # italic=True,
                                            style={"fontSize": 22},
                                        ),
                                    ]
                                )
                            ),
                            fac.AntdCol(
                                fac.AntdSpace(
                                    [
                                        fac.AntdFormItem(
                                            fac.AntdSwitch(
                                                id="enable-multi-round",
                                                checked=False,
                                                checkedChildren="开启",
                                                unCheckedChildren="关闭",
                                            ),
                                            label="多轮对话",
                                            style={"marginBottom": 0},
                                        ),
                                        fac.AntdTooltip(
                                            fac.AntdButton(
                                                id="export-history-qa-records",
                                                icon=fac.AntdIcon(icon="antd-save"),
                                                type="primary",
                                                shape="circle",
                                            ),
                                            title="导出当前全部对话记录",
                                        ),
                                        fac.AntdTooltip(
                                            fac.AntdButton(
                                                id="clear-exists-records",
                                                icon=fac.AntdIcon(icon="antd-clear"),
                                                type="primary",
                                                shape="circle",
                                                danger=True,
                                            ),
                                            title="一键清空当前对话",
                                        ),
                                    ]
                                )
                            ),
                        ],
                        justify="space-between",
                    ),
                    # 聊天记录容器
                    html.Div(
                        [
                            fac.AntdSpace(
                                [
                                    fac.AntdAvatar(
                                        mode="icon",
                                        icon="antd-robot",
                                        style={"background": "#1890ff"},
                                    ),
                                    fuc.FefferyDiv(
                                        fac.AntdText(
                                            "你好，欢迎使用基于 ChatGPT 服务的在线问答机器人。",
                                            style={"fontSize": 16},
                                        ),
                                        className="chat-record-container",
                                    ),
                                ],
                                align="start",
                                style={"padding": "10px 15px", "width": "100%"},
                            ),
                            fac.AntdSpace(
                                [
                                    fac.AntdAvatar(
                                        mode="icon",
                                        icon="antd-robot",
                                        style={"background": "#1890ff"},
                                    ),
                                    fuc.FefferyDiv(
                                        fmc.FefferyMarkdown(
                                            markdownStr=feature,
                                            style={
                                                "fontSize": 16,
                                                "fontFamily": 'Palatino, palatino linotype, palatino lt std, "思源宋体 CN", sans-serif',
                                            },
                                        ),
                                        className="chat-record-container",
                                    ),
                                ],
                                align="start",
                                style={"padding": "10px 15px", "width": "100%"},
                            ),
                            fac.AntdSpace(
                                [
                                    fac.AntdAvatar(
                                        mode="icon",
                                        icon="antd-robot",
                                        style={"background": "#1890ff"},
                                    ),
                                    fuc.FefferyDiv(
                                        fmc.FefferyMarkdown(
                                            markdownStr="请向我提问！我会在准备好回答后一次性回复你，请耐心等待😄",
                                            style={
                                                "fontSize": 16,
                                                "fontFamily": 'Palatino, palatino linotype, palatino lt std, "思源宋体 CN", sans-serif',
                                            },
                                        ),
                                        className="chat-record-container",
                                    ),
                                ],
                                align="start",
                                style={"padding": "10px 15px", "width": "100%"},
                            ),
                        ],
                        id="chat-records",
                    ),
                    # 聊天输入区
                    fac.AntdSpace(
                        [
                            fac.AntdInput(
                                id="new-question-input",
                                mode="text-area",
                                autoSize=False,
                                allowClear=True,
                                placeholder="请输入问题：",
                                size="large",
                                style={"fontSize": 16},
                            ),
                            fac.AntdButton(
                                "提交",
                                id="send-new-question",
                                type="primary",
                                block=True,
                                autoSpin=True,
                                loadingChildren="思考中",
                                size="large",
                            ),
                        ],
                        direction="vertical",
                        size=2,
                        style={"width": "100%"},
                    ),
                ],
                shadow="always-shadow",
                className="chat-wrapper",
            ),
            className="root-wrapper",
        ),
    ],
)


@app.callback(
    [
        Output("chat-records", "children"),
        Output("new-question-input", "value"),
        Output("send-new-question", "loading"),
        Output("response-status-message", "children"),
        Output("multi-round-store", "data"),
    ],
    [
        Input("send-new-question", "nClicks"),
        Input("clear-exists-records", "nClicks"),
        Input("enable-multi-round", "checked"),
    ],
    [
        State("new-question-input", "value"),
        State("chat-records", "children"),
        State("multi-round-store", "data"),
    ],
    prevent_initial_call=True,
)
def send_new_question(
    new_question_trigger,
    clear_records_trigger,
    enable_multi_round,
    question,
    origin_children,
    multi_round_store,
):
    """
    控制以渲染或清空对话框内容为目的的操作，包括处理新问题的发送、已有记录的清空、多轮对话模式的切换等
    """

    # 若当前回调由提交新问题触发
    if dash.ctx.triggered_id == "send-new-question" and new_question_trigger:

        # 检查问题输入是否有效
        if not question:
            return [
                dash.no_update,
                dash.no_update,
                False,
                fac.AntdMessage(content="请完善问题内容后再进行提交！", type="warning"),
                dash.no_update,
            ]

        # 尝试将当前的问题发送至 ChatGPT 问答服务接口
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=(
                    # 若当前模式为多轮对话模式，则附带上历史对话记录以维持对话上下文
                    [
                        *(multi_round_store.get("history") or []),
                        {"role": "user", "content": question},
                    ]
                    if enable_multi_round
                    else [{"role": "user", "content": question}]
                ),
                # 设置请求超时时长
                timeout=10,
            )

        except Exception as e:
            return [
                dash.no_update,
                dash.no_update,
                False,
                fac.AntdMessage(content="回复生成失败，错误原因：" + str(e), type="error"),
                dash.no_update,
            ]

        # 将上一次历史问答记录中 id 为 latest-response-begin 的元素过滤掉
        origin_children = [
            child
            for child in origin_children
            if child["props"].get("id") != "latest-response-begin"
        ]

        # 更新各输出目标属性
        return [
            [
                *origin_children,
                # 渲染当前问题
                fac.AntdSpace(
                    [
                        fac.AntdAvatar(
                            mode="text", text="我", style={"background": "#1890ff"}
                        ),
                        fuc.FefferyDiv(
                            fac.AntdText(
                                question, copyable=True, style={"fontSize": 16}
                            ),
                            className="chat-record-container",
                            style={"maxWidth": 680},
                        ),
                    ],
                    align="start",
                    style={
                        "padding": "10px 15px",
                        "width": "100%",
                        "flexDirection": "row-reverse",
                    },
                ),
                # 在当前问题回复之前注入辅助滚动动作的目标点
                html.Div(id="latest-response-begin"),
                # 渲染当前问题的回复
                fac.AntdSpace(
                    [
                        fac.AntdAvatar(
                            mode="icon",
                            icon="antd-robot",
                            style={"background": "#1890ff"},
                        ),
                        fuc.FefferyDiv(
                            fmc.FefferyMarkdown(
                                markdownStr=response["choices"][0]["message"][
                                    "content"
                                ],
                                codeTheme="okaidia",
                                codeFallBackLanguage="python",  # 遇到语言不明的代码块，统统视作 python 渲染
                                style={
                                    "fontFamily": 'Palatino, palatino linotype, palatino lt std, "思源宋体 CN", sans-serif',
                                },
                            ),
                            className="chat-record-container",
                            style={"maxWidth": 680},
                        ),
                    ],
                    align="start",
                    style={"padding": "10px 15px", "width": "100%"},
                ),
            ],
            None,
            False,
            [
                fac.AntdMessage(content="回复生成成功", type="success"),
                # 新的滚动动作
                fuc.FefferyScroll(
                    scrollTargetId="latest-response-begin",
                    scrollMode="target",
                    executeScroll=True,
                    containerId="chat-records",
                ),
            ],
            # 根据是否处于多轮对话模式选择返回的状态存储数据
            {
                "status": "开启" if enable_multi_round else "关闭",
                "history": [
                    *(multi_round_store.get("history") or []),
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": response["choices"][0]["message"]["content"],
                    },
                ],
            },
        ]

    # 若当前回调由清空记录按钮触发
    elif dash.ctx.triggered_id == "clear-exists-records" and clear_records_trigger:

        return [
            [origin_children[0]],
            None,
            False,
            fac.AntdMessage(content="已清空", type="success"),
            {"status": "开启" if enable_multi_round else "关闭", "history": []},
        ]

    # 若当前回调由多轮对话状态切换开关触发
    elif dash.ctx.triggered_id == "enable-multi-round":

        return [
            [origin_children[0]],
            None,
            False,
            fac.AntdMessage(
                content=("已开启多轮对话模式" if enable_multi_round else "已关闭多轮对话模式"),
                type="success",
            ),
            {"status": "开启" if enable_multi_round else "关闭", "history": []},
        ]

    return [dash.no_update, dash.no_update, False, None, dash.no_update]


@app.callback(
    Output("history-qa-records-download", "data"),
    Input("export-history-qa-records", "nClicks"),
    State("multi-round-store", "data"),
    prevent_initial_call=True,
)
def export_history_qa_records(nClicks, history_qa_records):
    """
    处理将当前全部对话记录导出为 markdown 文件的操作
    """

    if nClicks and history_qa_records.get("history"):

        # 拼接历史 QA 记录
        return_md_str = ""

        for record in history_qa_records["history"]:
            if record["role"] == "user":
                return_md_str += "\n#### 问题：{}\n".format(record["content"])

            else:
                return_md_str += "\n#### 回答：\n{}".format(record["content"])

        return dict(
            content=return_md_str,
            filename="问答记录{}.md".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        )


if __name__ == "__main__":
    app.run_server(debug=True)
