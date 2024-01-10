import streamlit
from . import ContentView
from pandas import DataFrame
from streamlit_pg.modules.openai_manager.openai_manager import OpenAIManager


class ModerationContentView(ContentView):
    def __init__(self, st: streamlit, oai_manager: OpenAIManager):
        super().__init__(st, oai_manager)
        self.openai_doc: str = "https://platform.openai.com/docs/guides/moderation/moderation"

    def view(self) -> None:
        self.st.write("OpenAI API Introduction [link](%s)" % self.openai_doc)
        self.st.header("[Request]")
        content: str = self.st.text_area("Type a Text to moderation")
        is_clicked: bool = self.st.button('Send Request')

        if not is_clicked:
            return

        self.st.header("[Response]")
        data: dict = self.oai_manager.moderation(content)

        if not data:
            return

        self.st.subheader("Category Scores")

        category_scores: dict = data.get("category_scores")
        print(category_scores)

        # DATA CHART
        chart_data: DataFrame = DataFrame(category_scores.values(), category_scores.keys())
        self.st.bar_chart(chart_data)
        print(chart_data)

        # DATA COLUMNS
        category_cnt: int = len(category_scores)
        #category_column_tuple: tuple = self.st.columns(category_cnt)

        for idx, (k, v) in enumerate(category_scores.items()):
            print("{:.9f}".format(v))
            #category_column: streamlit.delta_generator.DeltaGenerator = category_column_tuple[idx]
            #category_column.metric(label=k, value=round(v, 2))
            self.st.metric(label=k, value=v)

        # JSON DATA
        self.st.subheader("Response Data")
        self.st.json(data)
