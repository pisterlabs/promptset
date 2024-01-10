from pathlib import Path
from typing import TextIO

from langchain.chat_models import ChatOpenAI
from langchain.globals import set_verbose
from langchain.llms.openai import OpenAI
from megamock import MegaMock

from table_merger.table_mergers import ColumnInfo, TableMergeOperation, TableMergerManager

set_verbose(True)


class TestTableMergers:
    def test_ready(self, gpt3: OpenAI, example_template_csv: TextIO) -> None:
        # happy path
        tm = TableMergerManager(gpt3)
        assert tm.ready(template_file=example_template_csv) is True

        assert tm.template_columns
        assert not tm.errors

    def test_prep_csv_file_from_path(
        self, gpt3: OpenAI, example_template_csv: TextIO, table_a: Path
    ) -> None:
        tm = TableMergerManager(gpt3)
        tm.ready(template_file=example_template_csv)

        merge_operation = tm.prep_csv_file_from_path(table_a)
        assert not merge_operation.errors
        assert merge_operation.incoming_column_info
        assert not merge_operation.suggested_merge_info  # is not done here


class TestTableMergeOperation:
    def test_create_suggested_merge_info(
        self,
        template_column_info: list[ColumnInfo],
        incoming_column_info: list[ColumnInfo],
        gpt4: ChatOpenAI,
    ) -> None:
        merge_op = TableMergeOperation(
            template_column_info, incoming_column_info, MegaMock.it(TextIO)
        )
        column_merge_info = merge_op.create_suggested_merge_info(gpt4)
        assert merge_op.suggested_merge_info is column_merge_info
        assert not column_merge_info.errors
        assert column_merge_info.column_mapping
        assert len(column_merge_info.column_mapping) == len(template_column_info)

        template_column_names = {x.name for x in template_column_info}
        incoming_column_names = {x.name for x in incoming_column_info}

        assert (
            set([x.template_column for x in column_merge_info.column_mapping])
            == template_column_names
        )
        used_incoming_cols = set(
            [x.incoming_column for x in column_merge_info.column_mapping]
        ).intersection(incoming_column_names)
        assert len(used_incoming_cols) == len(template_column_names)

    def test_create_suggested_transformation_operations(self, gpt4: ChatOpenAI):
        # Sample mock data for template_column_info and incoming_column_info
        template_col_info = [
            ColumnInfo(
                name="date",
                type="string",
                output_format="%Y-%m-%d",
                empty_expected=False,
                example_values=["2023-10-18"],
            ),
            ColumnInfo(
                name="name",
                type="string",
                output_format="",
                empty_expected=False,
                example_values=["John Doe"],
            ),
        ]

        incoming_col_info = [
            ColumnInfo(
                name="incoming_date",
                type="string",
                output_format="",
                empty_expected=False,
                example_values=["18/10/2023"],
            ),
            ColumnInfo(
                name="incoming_name",
                type="string",
                output_format="",
                empty_expected=False,
                example_values=["Doe, John"],
            ),
        ]

        # Instantiate the TableMergeOperation object
        merge_op = TableMergeOperation(template_col_info, incoming_col_info, MegaMock.it(TextIO))

        # Mock actual_column_mapping to be used within the method
        merge_op.actual_column_mapping = {"date": "incoming_date", "name": "incoming_name"}

        # Call the method and get the result
        transformations = merge_op.create_suggested_transformation_operations(gpt4)

        # Assertions
        # Assert that there's a valid transformation for each column
        assert merge_op.suggested_transformation_operations is transformations
        assert len(transformations.transformations) == len(template_col_info)

        # Check that the names of template columns are covered in transformations
        template_column_names = {x.name for x in template_col_info}
        transformed_column_names = {x.column_name for x in transformations.transformations}
        assert template_column_names == transformed_column_names
