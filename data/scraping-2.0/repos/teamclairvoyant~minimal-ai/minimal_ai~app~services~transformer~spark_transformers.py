import asyncio
import logging
from typing import Any

from langchain.chat_models import ChatOpenAI
from pydantic.dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark_ai import SparkAI

from minimal_ai.app.services.minimal_exception import MinimalETLException
from minimal_ai.app.utils.spark_utils import DataframeUtils

logger = logging.getLogger(__name__)


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class SparkTransformer:
    current_task: Any
    spark: SparkSession

    async def transform(self) -> None:
        """method to execute the transformer

        Raises:
            MinimalETLException
        """
        match self.current_task.transformer_type:
            case "join":
                await self.join_df()
            case "sparkAI":
                await self.spark_ai_transform()
            # case "pivot":
            #     return await self.pivot()
            case "filter":
                return await self.filter_df()
            case _:
                logger.error('Transformer type - %s not supported',
                             self.current_task.transformer_type)
                raise MinimalETLException(
                    f'Transformer type - {self.current_task.transformer_type} not supported')

    async def filter_df(self) -> None:
        """method to filter the dataframe on the given condition"""
        try:
            logger.info("Loading data variables from upstream task")
            logger.info("Filtering dataframe on the condition provided")
            _df = await DataframeUtils.get_df_from_alias(self.spark,
                                                         self.current_task.upstream_tasks[0],
                                                         self.current_task.transformer_config['filter'])
            _df.createOrReplaceTempView(self.current_task.uuid)
            asyncio.create_task(self.current_task.pipeline.variable_manager.add_variable(
                self.current_task.pipeline.uuid,
                self.current_task.uuid,
                self.current_task.uuid,
                _df.toJSON().take(200)
            ))
        except Exception as excep:
            raise MinimalETLException(
                f'Failed to filter dataframe - {excep.args}')

    async def spark_ai_transform(self) -> None:
        """method to transform dataframe using sparkAI
        """
        try:
            logger.info("Activating SparkAI")

            spark_ai = SparkAI(llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),   # type: ignore
                               spark_session=self.spark)
            spark_ai.activate()

            _df = await DataframeUtils.get_df_from_alias(self.spark, self.current_task.upstream_tasks[0])

            _df_ai = spark_ai.transform_df(
                df=_df, desc=self.current_task.transformer_config["prompt"])

            _df_ai.createOrReplaceTempView(self.current_task.uuid)

            asyncio.create_task(self.current_task.pipeline.variable_manager.add_variable(
                self.current_task.pipeline.uuid,
                self.current_task.uuid,
                self.current_task.uuid,
                _df_ai.toJSON().take(200)
            ))
        except Exception as excep:
            raise MinimalETLException(
                f'Failed to transform dataframe with spark AI - {excep.args}')

    async def join_df(self) -> None:
        """
        Method to join dataframes from two tasks
        """
        try:
            logger.info('Loading data variables from upstream tasks')

            left_df = await DataframeUtils.get_df_from_alias(self.spark,
                                                             self.current_task.transformer_config['left_table'])

            right_df = await DataframeUtils.get_df_from_alias(self.spark,
                                                              self.current_task.transformer_config['right_table'])

            logger.info('Data loaded')
            logger.info('modifying duplicate column names')
            left_on = self.current_task.transformer_config['left_on']
            right_on = self.current_task.transformer_config['right_on']

            for index in range(len(left_on)):
                if left_on[index] == right_on[index]:
                    left_df = left_df.withColumnRenamed(
                        left_on[index],
                        f"{left_on[index]}_{self.current_task.transformer_config['left_table']}")
                    right_df = right_df.withColumnRenamed(
                        right_on[index],
                        f"{right_on[index]}_{self.current_task.transformer_config['right_table']}")
                    left_on[
                        index] = f"{left_on[index]}_{self.current_task.transformer_config['left_table']}"
                    right_on[
                        index] = f"{right_on[index]}_{self.current_task.transformer_config['right_table']}"

            on = [
                col(f) == col(s)
                for (f, s) in zip(self.current_task.transformer_config['left_on'],
                                  self.current_task.transformer_config['right_on'])
            ]

            _df = left_df.join(
                right_df, on=on, how=self.current_task.transformer_config['how'])

            _df.createOrReplaceTempView(self.current_task.uuid)

            asyncio.create_task(self.current_task.pipeline.variable_manager.add_variable(
                self.current_task.pipeline.uuid,
                self.current_task.uuid,
                self.current_task.uuid,
                _df.toJSON().take(200)
            ))
        except Exception as excep:
            logger.info(excep)
            raise MinimalETLException(
                f'Failed to join dataframe - {excep.args}')
