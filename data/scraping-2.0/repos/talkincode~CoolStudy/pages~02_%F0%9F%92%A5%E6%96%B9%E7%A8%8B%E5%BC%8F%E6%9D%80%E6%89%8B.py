import streamlit as st
import sympy as sp
import os
import openai
from openai import OpenAI


# 设置侧边栏
st.sidebar.markdown("# 💥方程式杀手")

# 获取环境变量中的 OpenAI API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 设置 OpenAI API 密钥

client = OpenAI()


# 定义一个函数来发送请求到 OpenAI API
@st.cache_data
def get_inference(equation_str, solution_str):
    try:
        messages = [{
            "role": "system",
            "content": "请你充当一个数学老师， 我会提供给你一个方程式和它的解，"
                       "你需要提供给我解释这个方程式的求解过程，优化策略如下：\n\n"
                       "- 你需要分析并解释这个方程式的求解过程\n"
                       "- 你需要尽量使用简单的语言来解释\n"
                       "- 请将公式使用$括起来\n\n"
        }, {
            "role": "user",
            "content": f"解释以下方程的求解过程,让我们一步一步分析:\n\n"
                       f"方程: {equation_str}\n\n"
                       f"解: {solution_str}\n\n"
                       f"解释:"
        }]
        print(messages)
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0,
            max_tokens=2000,
        )
        print(response)
        result = response.choices[0].message.content
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e)


if 'chat_running' not in st.session_state:
    st.session_state['chat_running'] = False


# 设置标题和副标题
st.title("💥方程式杀手")
st.subheader("一个简单的工具，用于化简和解决方程式")
st.markdown("---")

# 创建三个并排的列
lcol, mcol, rcol = st.columns([4, 1, 4])

# 在左侧列创建左侧表达式输入框
left_expr = lcol.text_input("方程左侧表达式,比如（2 * x^2）", "a*x**2 + b*x")

# 在中间列显示等号，并调整位置
mcol.markdown("<h3 style='text-align: center; margin-top: 20px;'>=</h3>", unsafe_allow_html=True)

# 在右侧列创建右侧表达式输入框
right_expr = rcol.text_input("方程右侧表达式", "c")

# 创建一个输入框来输入变量
vars_input = st.text_input("变量 (多个变量使用空格分隔, 比如 x y)", "x")

# 所有变量
symbols = sp.symbols(vars_input)

# 解析表达式
try:
    equation = sp.sympify(left_expr + "-" + right_expr,  evaluate=False)
except Exception as e:
    equation = None
    st.error(f"无效的表达式，请检查输入的表达式是否正确, {e}")

# 显示完整的方程式
st.write("完整的方程式：")
st.latex(sp.latex(equation))
st.divider()

# 创建并排的按钮
col1, col2, col3 = st.columns(3)

if col1.button("化简"):
    simplified_expr = sp.simplify(equation)
    st.write("化简结果:")
    st.latex(sp.latex(simplified_expr))

if col2.button("解答"):
    solutions = sp.solve(equation, symbols)
    st.write("解答结果:")
    for solution in solutions:
        st.latex(sp.latex(solution))

if col3.button("推理", disabled=st.session_state['chat_running']):
    # 创建方程式和解的字符串表示
    st.session_state['chat_running'] = True
    with st.spinner("Thinking..."):
        solutions = sp.solve(equation, symbols)
        equation_str = f"{left_expr} = {right_expr}"
        solution_str = ', '.join([sp.latex(sol) for sol in solutions])

        # 获取推理结果
        result = get_inference(equation_str, solution_str)
        st.write("推理结果:")
        st.markdown("---")
        st.markdown(result)
        st.session_state['chat_running'] = False


if st.session_state['chat_running']:
    st.write("推理任务正在运行中，请等待...")
