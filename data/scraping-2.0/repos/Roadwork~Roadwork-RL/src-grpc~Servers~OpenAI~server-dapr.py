import sys
import os
import logging
import time
from concurrent import futures
from datetime import datetime

import grpc

# Dapr Libraries
from dapr.proto.common.v1 import common_pb2 as commonv1pb
from dapr.proto.dapr.v1 import dapr_pb2 as dapr_messages
from dapr.proto.dapr.v1 import dapr_pb2_grpc as dapr_services
from dapr.proto.daprclient.v1 import daprclient_pb2 as daprclient_messages
from dapr.proto.daprclient.v1 import daprclient_pb2_grpc as daprclient_services

# Custom Protobuf
import proto_compiled.roadwork_pb2 as roadwork_messages

import protobuf_helpers

from google.protobuf.any_pb2 import Any

# Import OpenAI
from OpenAIEnv import Envs

APP_PORT_GRPC  = os.getenv('APP_GRPC_PORT',  50050)
DAPR_PORT_HTTP = os.getenv('DAPR_HTTP_PORT', 3500)
DAPR_PORT_GRPC = os.getenv('DAPR_GRPC_PORT', 50001) # Note: currently 50001 is always default

print(f"==================================================")
print(f"DAPR_PORT_GRPC: {DAPR_PORT_GRPC}; DAPR_PORT_HTTP: {DAPR_PORT_HTTP}")
print(f"APP_PORT_GRPC: {APP_PORT_GRPC}")
print(f"==================================================")

# import gym
envs = Envs()

# # Start a gRPC client
channel = grpc.insecure_channel(f"localhost:{DAPR_PORT_GRPC}")
client = dapr_services.DaprStub(channel)
print(f"Started gRPC client on DAPR_GRPC_PORT: {DAPR_PORT_GRPC}")

# Our server methods
class DaprClientServicer(daprclient_services.DaprClientServicer):
    def OnInvoke(self, request, context):
        res = ""

        if request.method == 'create':
            req = protobuf_helpers.from_any_pb(roadwork_messages.CreateRequest, request.data)
            res = roadwork_messages.CreateResponse(instanceId=envs.create(req.envId))
            res = protobuf_helpers.to_any_pb(res)
        elif request.method == 'reset':
            req = protobuf_helpers.from_any_pb(roadwork_messages.ResetRequest, request.data)
            res = roadwork_messages.ResetResponse(observation=envs.reset(req.instanceId))
            res = protobuf_helpers.to_any_pb(res)
        elif request.method == 'action-space-sample':
            req = protobuf_helpers.from_any_pb(roadwork_messages.ActionSpaceSampleRequest, request.data)
            res = roadwork_messages.ActionSpaceSampleResponse(action=envs.get_action_space_sample(req.instanceId))
            res = protobuf_helpers.to_any_pb(res)
        elif request.method == 'action-space-info':
            req = protobuf_helpers.from_any_pb(roadwork_messages.ActionSpaceInfoRequest, request.data)
            res = roadwork_messages.ActionSpaceInfoResponse(result=envs.get_action_space_info(req.instanceId))
            res = protobuf_helpers.to_any_pb(res)
        elif request.method == 'observation-space-info':
            req = protobuf_helpers.from_any_pb(roadwork_messages.ObservationSpaceInfoRequest, request.data)
            res = roadwork_messages.ObservationSpaceInfoResponse(result=envs.get_observation_space_info(req.instanceId))
            res = protobuf_helpers.to_any_pb(res)
        elif request.method == 'step':
            req = protobuf_helpers.from_any_pb(roadwork_messages.StepRequest, request.data)
            res_step = envs.step(req.instanceId, req.action, req.render) # Returns 0 = obs_jsonable, 1 = reward, 2 = done, 3 = info in array

            # Observation Space
            res_osi = envs.get_observation_space_info(req.instanceId)
            space_wrapper = roadwork_messages.SpaceWrapper()

            if res_osi.HasField('discrete'):
                space_discrete = roadwork_messages.SpaceDiscrete()
                space_discrete.observation = res_step[0]
                space_wrapper.discrete.CopyFrom(space_discrete)
            elif res_osi.HasField('box'):
                space_box = roadwork_messages.SpaceBox()
                space_box.observation.extend(res_step[0])
                space_wrapper.box.CopyFrom(space_box)
            else:
                logging.error("Unsupported Space Type: %s" % res_step[3]['name'])
                logging.error(info)

            res = roadwork_messages.StepResponse(reward=res_step[1], isDone=res_step[2], info=res_step[3], observation=space_wrapper)
            res = protobuf_helpers.to_any_pb(res)
        elif request.method == 'monitor-start':
            req = protobuf_helpers.from_any_pb(roadwork_messages.BaseRequest, request.data)
            envs.monitor_start(req.instanceId, '/mnt/output-server', True, False, 10) # Log to local dir so we can reach it
            res = roadwork_messages.BaseResponse()
            res = protobuf_helpers.to_any_pb(res)
        elif request.method == 'monitor-stop':
            req = protobuf_helpers.from_any_pb(roadwork_messages.BaseRequest, request.data)
            envs.monitor_close(req.instanceId)
            res = roadwork_messages.BaseResponse()
            res = protobuf_helpers.to_any_pb(res)
        else:
            res = Any(value='METHOD_NOT_SUPPORTED'.encode('utf-8'))

        # print(f"[OnInvoke][{request.method}] Done @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Return response to caller
        content_type = "text/plain; charset=UTF-8"
        return commonv1pb.InvokeResponse(data=res, content_type=content_type)

# Create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10))
daprclient_services.add_DaprClientServicer_to_server(DaprClientServicer(), server)

# Start the gRPC server
print(f'Starting server. Listening on port {APP_PORT_GRPC}.')
server.add_insecure_port(f'[::]:{APP_PORT_GRPC}')
server.start()

# Since server.start() doesn't block, we need to do a sleep loop
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)