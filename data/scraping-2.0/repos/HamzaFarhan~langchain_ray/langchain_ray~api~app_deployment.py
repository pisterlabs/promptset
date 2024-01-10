from langchain_ray.api.app import *

params = dict(
    block_size=2,
    num_cpus=4,
    num_gpus=0.2,
)

deployment_handle = TNetIngress.bind(**params)
