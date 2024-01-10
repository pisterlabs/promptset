import sys
import os
import torch
import openai
import argparse
import logging
from langchain import OpenAI
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt, LLMPredictor, PromptHelper, ServiceContext
from gpt_index import GPTListIndex, SimpleDirectoryReader
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.embeddings.langchain import LangchainEmbedding
from transformers import pipeline
from IPython.display import Markdown, display
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompt_type import PromptType


class TextToClusterDefinitionPrompt(Prompt):
    prompt_type: PromptType = PromptType.CLUSTER_DEFINITION
    input_variables: list[str] = ["query_str", "schema"]

QDRANT_CLUSTERDEFINTION_SCHEMA =  '''apiVersion: apps.kubeblocks.io/v1alpha1
kind: ClusterDefinition
metadata:
  name: qdrant-standalone
  labels:
    {{- include "qdrant.labels" . | nindent 4 }}
spec:
  type: qdrant
  connectionCredential:
    username: root
    password: "$(RANDOM_PASSWD)"
    endpoint: "$(SVC_FQDN):$(SVC_PORT_tcp-qdrant)"
    host: "$(SVC_FQDN)"
    port: "$(SVC_PORT_tcp-qdrant)"
  componentDefs:
    - name: qdrant
      workloadType: Stateful
      characterType: qdrant
      probes:
      monitor:
        builtIn: false
        exporterConfig:
          scrapePath: /metrics
          scrapePort: 9187
      logConfigs:
      configSpecs:
        - name: qdrant-standalone-config-template
          templateRef: qdrant-standalone-config-template
          volumeName: qdrant-config
          namespace: {{ .Release.Namespace }}
      service:
        ports:
          - name: tcp-qdrant
            port: 6333
            targetPort: tcp-qdrant
          - name: grpc-qdrant
            port: 6334
            targetPort: grpc-qdrant
      volumeTypes:
        - name: data
          type: data
      podSpec:
        securityContext:
          fsGroup: 1001
        containers:
          - name: qdrant
            imagePullPolicy: {{default .Values.images.pullPolicy "IfNotPresent"}}
            securityContext:
              runAsUser: 0
            livenessProbe:
              failureThreshold: 3
              httpGet:
                path: /
                port: 6333
                scheme: HTTP
              periodSeconds: 15
              successThreshold: 1
              timeoutSeconds: 10
            readinessProbe:
              failureThreshold: 2
              httpGet:
                path: /
                port: 6333
                scheme: HTTP
              initialDelaySeconds: 5
              periodSeconds: 15
              successThreshold: 1
              timeoutSeconds: 3
            startupProbe:
              failureThreshold: 18
              httpGet:
                path: /
                port: 6333
                scheme: HTTP
              periodSeconds: 10
              successThreshold: 1
              timeoutSeconds: 3
            terminationMessagePath: /dev/termination-log
            terminationMessagePolicy: File
            volumeMounts:
              - mountPath: /qdrant/config/
                name: qdrant-config
              - mountPath: /qdrant/storage
                name: data
            dnsPolicy: ClusterFirst
            enableServiceLinks: true
            ports:
              - name: tcp-qdrant
                containerPort: 6333
              - name: grpc-qdrant
                containerPort: 6334
              - name: tcp-metrics
                containerPort: 9091
            command:
              - ./qdrant
            env:'''

DEFAULT_TEXT_TO_CD_TMPL = (
    "Given an input question, generate the answer in YAML format."
    "The cluster definition YAML is a dialect in YAML format. "
    "A cluster definition YAML has fixed schema, or specification."
    "Cluster defintion declares the components of a database cluster instance."
    "Each component has a defintion in componentDef spec. "
    "The major part of a component defintion is k8s podSpec. \n"
    "Here we give an example of generating a cluster definition YAML: "
    "Question: Generate a qdrant cluster defintion for me, "
    "the fields in the spec can be the default values of qdrant."
    "Answer: the qdrant clusterdefintion YAML is {schema}\n"
    "Use the following format:\n"
    "Question: Question here\n"
    "Answer: Final answer here\n"
    "Question: {query_str}\n"
)

DEFAULT_TEXT_TO_CD_PROMPT = TextToClusterDefinitionPrompt(
    #DEFAULT_TEXT_TO_CD_TMPL, stop_token="\nResult:"
    DEFAULT_TEXT_TO_CD_TMPL
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query Engine for KubeBlocks.")
    parser.add_argument("query_str", type=str, help="Query string for ask.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    query_str = args.query_str
    print("query:", query_str)

    # set env for OpenAI api key
    # os.environ['OPENAI_API_KEY'] = ""

    # set log level
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))# define LLM

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=3000))

    # define prompt helper
    # set maximum input size
    max_input_size = 32768
    # set number of output tokens
    num_output = 32768
    # set maximum chunk overlap
    max_chunk_overlap = 200
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    response, s = service_context.llm_predictor.predict (
        DEFAULT_TEXT_TO_CD_PROMPT,
        query_str=query_str,
        schema=QDRANT_CLUSTERDEFINTION_SCHEMA,
    )

    # print(s)
    print(response)


if __name__ == "__main__":
    main()

