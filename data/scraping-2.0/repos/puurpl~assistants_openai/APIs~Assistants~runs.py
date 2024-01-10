from openai import OpenAI


async def create_run(thread_id, assistant_id):
    client = OpenAI()
    run = await client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assistant_id,
    )
    return(run)

async def list_runs(thread_id, order, limit):
    client = OpenAI()
    runs = await client.beta.threads.runs.list(
    thread_id=thread_id,
    order=order,
    limit=limit,
    )
    return(runs)

async def retrieve_run(thread_id, run_id):
    client = OpenAI()
    run = await client.beta.threads.runs.retrieve(
    thread_id=thread_id,
    run_id=run_id,
    )
    return(run)

async def modify_run(thread_id, run_id, metadata):
    client = OpenAI()
    run = await client.beta.threads.runs.update(
    thread_id=thread_id,
    run_id=run_id,
    metadata=metadata, # metadata is a map
    )
    return(run)

async def cancel_run(thread_id, run_id):
    client = OpenAI()
    run = await client.beta.threads.runs.cancel(
    thread_id=thread_id,
    run_id=run_id,
    )
    return(run)

async def list_run_steps(thread_id, run_id, order, limit):
    client = OpenAI()
    run_steps = await client.beta.threads.runs.steps.list(
    thread_id=thread_id,
    run_id=run_id,
    order=order,
    limit=limit,
    )
    return(run_steps)

async def retrieve_run_step(thread_id, run_id, step_id):
    client = OpenAI()
    run_step = await client.beta.threads.runs.steps.retrieve(
    thread_id=thread_id,
    run_id=run_id,
    step_id=step_id,
    )
    return(run_step)

async def create_thread_and_run(assistant_id, thread):
    client = OpenAI()
    thread_and_run = await client.beta.threads.create(
    assistant_id=assistant_id,
    thread=thread,
    )
    return(thread_and_run)









