import json

import altair as alt
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from prompts.visualizer import VISUALIZATION_PICKER_PROMPT


def get_suggested_visualization_response_from_ai(
        sql_query, user_question):

    model_name = 'gpt-4'
    chat_api = ChatOpenAI(
        temperature=0, streaming=True,
        model_name=model_name
    )

    messages = [
        SystemMessage(content=VISUALIZATION_PICKER_PROMPT),
        HumanMessage(
            content=f"""
                user_query: {sql_query}
                sql_query: {user_question}
            """.strip()
        )
    ]

    visualization_response = chat_api(messages).content
    visualization_response = json.loads(visualization_response)

    return visualization_response


def plot_visualization_for_user_question(
        visualization_response, query_output_df):

    # plotting the visualization based on type
    visualization_type = visualization_response["visualization"]

    if visualization_type == "bar_chart":
        _plot_bar_chart(visualization_response, query_output_df)

    elif visualization_type == "line_chart":
        _plot_line_chart(visualization_response, query_output_df)

    elif visualization_type == "pie_chart":
        _plot_pie_chart(visualization_response, query_output_df)

    elif visualization_type == "stacked_bar_chart":
        _plot_stacked_bar_chart(visualization_response, query_output_df)

    elif visualization_type == "not_required" or "not_sure":
        pass

    else:
        print("visualization type not found: ", visualization_type)


def _plot_bar_chart(visualization_response, query_output_df):
    horizontal_chart = alt.Chart(query_output_df).mark_bar().encode(
        y=alt.Y(visualization_response['x_axis_column_name'] + ':N',
                title=visualization_response['x_axis_column_name']),
        x=alt.X(visualization_response['y_axis_column_name'] + ':Q',
                title=visualization_response['y_axis_column_name']),
        color=alt.Color(visualization_response['x_axis_column_name'],
                        legend=None)
    ).properties(
        title=f'{visualization_response["y_axis_column_name"]} vs '
              f'{visualization_response["x_axis_column_name"]}',
        height=500
    )

    st.altair_chart(horizontal_chart, use_container_width=True)


def _plot_line_chart(visualization_response, query_output_df):
    query_output_df.set_index(
        visualization_response['x_axis_column_name'], inplace=True)
    st.line_chart(query_output_df)


def _plot_pie_chart(visualization_response, query_output_df):
    base = alt.Chart(query_output_df).encode(
        alt.Theta(visualization_response['value_column_name'] + ":Q",
                  stack=True),
        alt.Color(visualization_response['labels_column_name'] + ":N")
    )

    donut = base.mark_arc(innerRadius=50, outerRadius=120)
    text = base.mark_text(radius=150, align='center', baseline='middle',
                          fontSize=15, fontWeight='bold').encode(
        text=alt.Text(visualization_response['value_column_name'] + ":Q"))
    chart = donut + text

    st.altair_chart(chart, use_container_width=True)


def _plot_stacked_bar_chart(visualization_response, query_output_df):
    id_vars = [
        col for col in visualization_response['columns']
        if col not in visualization_response['y_axis_columns']
    ]
    query_output_df = query_output_df.melt(
        id_vars=id_vars,
        value_vars=visualization_response['y_axis_columns'],
        var_name='variable',
        value_name='value'
    )
    chart = alt.Chart(query_output_df).mark_bar().encode(
        x=alt.X(f"{visualization_response['x_axis_columns'][0]}:N",
                title=visualization_response['x_axis_columns'][0].
                replace("_", " ").title()),
        y=alt.Y('value:Q',
                title=visualization_response['y_axis_columns'][0].
                replace("_", " ").title()),
        color=alt.Color(f"{visualization_response['color_columns'][0]}:N",
                        title=visualization_response['color_columns'][
                            0].replace("_", " ").title()),
        tooltip=[
            f"{col}:N"
            if col in visualization_response['x_axis_columns'] +
            visualization_response['color_columns'] else f"{col}:Q"
            for col in visualization_response['tooltip_columns']
        ],
        order=alt.Order('value:Q',
                        sort=visualization_response['order_type'])). \
        properties(width=alt.Step(50))

    st.altair_chart(chart, use_container_width=True)
