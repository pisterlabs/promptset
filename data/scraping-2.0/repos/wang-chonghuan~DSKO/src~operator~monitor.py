import os
import datetime
import json
import pykube
import requests
import openai
from states import CustomContext
import states
from configs import *

# The Prometheus server can be accessed via port 80 on the following DNS name from within your cluster: prometheus-server.default.svc.cluster.local
def get_prometheus_metric_in_cluster(logger):
    try:
        api = pykube.HTTPClient(pykube.KubeConfig.from_file())
        nodes = list(pykube.Node.objects(api).filter())
        node_ip = nodes[0].obj['status']['addresses'][0]['address']
        logger.info(f'SPOK_LOG call get_prometheus_metrics.............{node_ip}')

        query = 'sum(container_memory_usage_bytes{namespace="default", container="postgres"}) / sum(container_spec_memory_limit_bytes{namespace="default", container="postgres"})'
        response = requests.get(URL_PROMETHEUS_SERVER_IN_CLUSTER, params={'query': query})
        
        results = response.json()
        if results['status'] == 'success':
            logger.info('SPOK_LOG call success')
            metrics = results['data']['result']
            logger.info(json.dumps(metrics, indent=4))
            return metrics
        else:
            logger.info('SPOK_LOG Failed to query Prometheus: ignoring metric collection.')
    except Exception as e:
        logger.info(f'SPOK_LOG Error accessing Prometheus or processing results: {e}. Ignoring metric collection.')


def url_prometheus_server_out_cluster():
    api = pykube.HTTPClient(pykube.KubeConfig.from_file())
    nodes = list(pykube.Node.objects(api).filter())
    node_ip = nodes[0].obj['status']['addresses'][0]['address']
    url = f"http://{node_ip}:30090/api/v1/query"
    return url

def get_ai_advice(logger, message):
    # Load your API key from an environment variable or secret management service
    logger.info(f'SPOK_LOG openAI gpt4, sending message')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[
            {"role": "user", "content": message}
    ])
    logger.info('SPOK_LOG ' + chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def scale_on_metrics(api, logger, memo: CustomContext, spok_name, spok_ns):
    try:
        url = url_prometheus_server_out_cluster()
        logger.info(f'SPOK_LOG call get_prometheus_metrics on url out cluster.............{url}')

        queries = {
            "node_cpu_utilization_percentage": 'avg (rate(node_cpu_seconds_total{mode!="idle"}[5m])) * 100',
            "node_memory_utilization_percentage": 'avg((node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100)',
            "postgres_pod_cpu_usage_seconds_total": 'sum(rate(container_cpu_usage_seconds_total{namespace="default", container="postgres"}[5m]))',
            "postgres_pod_memory_usage_percentage": 'sum(container_memory_usage_bytes{namespace="default", container="postgres"} / container_spec_memory_limit_bytes{namespace="default", container="postgres"}) * 100'
        }

        results = {}
        for key, query in queries.items():
            response = requests.get(url, params={'query': query})
            
            if response.status_code == 200:
                result = response.json()['data']['result']

                # Convert timestamp and calculate averages if necessary
                for res in result:
                    timestamp = float(res['value'][0])
                    dt_object = datetime.datetime.fromtimestamp(timestamp)
                    formatted_time = dt_object.strftime('%Y%m%d%H%M%S')
                    res['value'][0] = formatted_time

                results[key] = result
            else:
                logger.info(f'SPOK_LOG Query failed: {query}')
                results[key] = None

        results["current_standby_replicas"] = memo.current_standby_replicas
        #results["postgres_pod_memory_usage_percentage"] = 99
        metrics = json.dumps(results, indent=4)
        logger.info(metrics)
        
        prompt = """
                    Please analyze the load information provided and decide how to scale the PostgreSQL cluster, 
                    considering whether to scale in or scale out. 
                    Remember that the PostgreSQL cluster must have a minimum of 1 master and 1 replica, and a maximum of 3 replicas. 
                    Then, complete the JSON below accordingly:
                    ```json
                    {
                        "description": "<A brief description of the current load situation and the recommended scaling decision>",
                        "desired_standby_replicas": "<The target number of replicas for the PostgreSQL cluster, the range is strictly 1,2,3>",
                        "alarm": "<A warning message if the scaling decision exceeds the cluster's capabilities>"
                    }
                    ```

                    Please return only the filled JSON, nothing more.
                """

        advice_str = get_ai_advice(logger, metrics + prompt)
        try:
            advice_json = json.loads(advice_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        logger.info('SPOK_LOG gpt4 call end')
        new_replicas = 1 #int(advice_json["desired_standby_replicas"])
        if new_replicas > 3 or new_replicas < 1:
            logger.error(f"wrong range")
            return
        states.update_spok_instance(spok_name, new_replicas)

    except Exception as e:
        logger.info(f'SPOK_LOG Error accessing Prometheus or processing results: {e}. Ignoring metric collection.')