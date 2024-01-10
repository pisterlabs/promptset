from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models.param import Param
from datetime import timedelta
from pathlib import Path

import sys
sys.path.append('/opt/airflow/common_package/')

from openai_gpt import OpenAIGPT
from aws_s3_bucket import AWSS3Download
from audio_transcribe import AudioTranscribe


aws_cloud = AWSS3Download()
audio_transcribe = AudioTranscribe()
open_ai_gpt = OpenAIGPT()



# %%
dag = DAG(
    dag_id="batch",
    schedule="0 3 * * *",   # https://crontab.guru/
    start_date=days_ago(0),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60),
    tags=["damg7245"],
)


with dag:


    get_all_batch_audio_files_from_s3 = PythonOperator(
        task_id='get_all_batch_audio_files_from_s3',
        python_callable= aws_cloud.get_all_batch_files,
        provide_context=True,
        do_xcom_push=True,
        dag=dag,
    )


    transcribe_all_batch_audio = PythonOperator(
        task_id='transcribe_all_batch_audio',
        python_callable= audio_transcribe.transcribe_batch_audio_link,
        op_kwargs={"audio_file_urls_string": "{{ ti.xcom_pull(task_ids='get_all_batch_audio_files_from_s3') }}"},
        provide_context=True,
        do_xcom_push=True,
        dag=dag,
    )



    moving_all_transcription_to_aws_bucket = PythonOperator(
        task_id='moving_all_transcription_to_aws_bucket',
        python_callable= aws_cloud.move_batch_audio_with_transcription,
        op_kwargs={"audio_file_with_transcribe": "{{ ti.xcom_pull(task_ids='transcribe_all_batch_audio') }}"},
        do_xcom_push=True,
        provide_context=True,
        dag=dag,
    )


    moving_all_audio_file_to_proccessd_aws_bucket = PythonOperator(
        task_id='moving_audio_file_to_proccessd_aws_bucket',
        python_callable= aws_cloud.move_batch_audio_to_processed_folder,
        op_kwargs={"audio_file_with_transcribe": "{{ ti.xcom_pull(task_ids='transcribe_all_batch_audio') }}"},
        provide_context=True,
        do_xcom_push=True,
        dag=dag,
    )


    generate_default_questions_for_batch_transcription = PythonOperator(
        task_id='generate_default_questions_for_batch_transcription',
        python_callable= open_ai_gpt.generate_questions_for_batch_transcribed_text,
        op_kwargs={"audio_file_with_transcribe": "{{ ti.xcom_pull(task_ids='transcribe_all_batch_audio') }}"},
        provide_context=True,
        do_xcom_push=True,
        dag=dag,
    )


    get_all_batch_audio_files_from_s3 >> transcribe_all_batch_audio >> [moving_all_transcription_to_aws_bucket, moving_all_audio_file_to_proccessd_aws_bucket] >> generate_default_questions_for_batch_transcription