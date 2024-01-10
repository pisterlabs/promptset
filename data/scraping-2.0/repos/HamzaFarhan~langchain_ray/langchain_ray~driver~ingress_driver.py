from langchain_ray.imports import *
from langchain_ray.utils import *
from langchain_ray.remote_utils import *
from ray.util.queue import Queue, Empty
from langchain_ray.driver import redis_kv_store


def get_task_from_kv_store(task_id, kv_store):
    task = kv_store.get(task_id)
    if task is None:
        raise Exception(f"No task entry found for task_id: {task_id}.")
    if type(task) != dict:
        raise Exception(f"Wrong type for task with task_id: {task_id}.")
    if len(task) == 0:
        raise Exception(f"Empty dict for task_id: {task_id}.")
    return task


# async def bulk_execute(data, kv_store):
#     task_id = data.get("task_id", "task_123")
#     chain = data["chain"]
#     chain_data = data["chain_data"]
#     try:
#         task = get_task_from_kv_store(task_id, kv_store)
#     except Exception as e:
#         raise Exception(e)
#     task["status"] = "TASK_STATUS_INPROGRESS"
#     try:
#         kv_store.insert(task_id, task)
#         res = chain(chain_data)
#         try:
#             task = get_task_from_kv_store(task_id, kv_store)
#         except Exception as e:
#             raise Exception(e)
#         task["status"] = "TASK_STATUS_FINISHED"
#         # task["results"] = json.dumps(res)
#         msg.info(f"Inserting task_id: {task_id}.", spaced=True)
#         kv_store.insert(task_id, task)
#         msg.info(f"Inserted task_id: {task_id}.", spaced=True)
#     except Exception as e:
#         error = f"bulk_execute failed with error: {e}"
#         msg.fail(error, spaced=True)
#         try:
#             task = get_task_from_kv_store(task_id, kv_store)
#         except Exception as e:
#             raise Exception(e)
#         task["status"] = "TASK_STATUS_ERROR"
#         task["error"] = error
#         kv_store.insert(task_id, task)


class Ingress:
    def __init__(
        self,
        redis_host="127.0.0.1",
        redis_port=6379,
    ):
        msg.info(f"Ingress RAY RESOURCES: {ray.available_resources()}", spaced=True)
        try:
            self.kv_store = redis_kv_store.KeyValueStore(
                redis_host=redis_host, redis_port=redis_port
            )
        except Exception as e:
            msg.fail(f"Ingress init failed with error: {e}", spaced=True)

    def bulk_execute(self, data, chain_creator):
        task_id = data.get("task_id", "task_123")
        # chain = data["chain"]
        chain_data = data["chain_data"]
        try:
            chain = chain_creator(chain_data)
        except Exception as e:
            msg.fail("Failed to create Chain.", spaced=True)
            raise Exception(e)
        try:
            task = get_task_from_kv_store(task_id, self.kv_store)
        except Exception as e:
            raise Exception(e)
        task["status"] = "TASK_STATUS_INPROGRESS"
        try:
            self.kv_store.insert(task_id, task)
            try:
                chain(chain_data, return_only_outputs=True)
            except Exception as e:
                msg.fail("Failed to run Chain.", spaced=True)
                raise Exception(e)
            try:
                task = get_task_from_kv_store(task_id, self.kv_store)
            except Exception as e:
                raise Exception(e)
            task["status"] = "TASK_STATUS_FINISHED"
            # task["results"] = json.dumps(res)
            msg.info(f"Inserting task_id: {task_id}.", spaced=True)
            self.kv_store.insert(task_id, task)
            msg.info(f"Inserted task_id: {task_id}.", spaced=True)
            del chain
        except Exception as e:
            error = f"bulk_execute failed with error: {e}"
            msg.fail(error, spaced=True)
            try:
                task = get_task_from_kv_store(task_id, self.kv_store)
            except Exception as e:
                raise Exception(e)
            task["status"] = "TASK_STATUS_ERROR"
            task["error"] = error
            self.kv_store.insert(task_id, task)

    def bulk_action(self, data, background_tasks, chain_creator):
        chain_data = data["chain_data"]
        task_data = {"chain_data": chain_data}
        try:
            task_id = gen_random_string(16)
            tenant_id = chain_data.get("tenant_id", "tenant_123")
            self.kv_store.insert(task_id, {"status": "TASK_STATUS_CREATED"})
            task_data["task_id"] = task_id
            task_data["tenant_id"] = tenant_id
            background_tasks.add_task(
                self.bulk_execute, task_data, chain_creator=chain_creator
            )
            # background_tasks.add_task(bulk_execute, task_data, self.kv_store)
            return {"task_id": task_id, "tenant_id": tenant_id}
        except Exception as e:
            raise Exception(f"Initiating Task failed with error: {e}")
