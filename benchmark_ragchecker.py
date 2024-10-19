from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import asyncio
from ollama import AsyncClient
import time
from threading import Thread
import sys

start_time = time.time()
client = AsyncClient(host="prog3.student.famnit.upr.si:6666")
# client = AsyncClient()

data = "data_to_bm.json"
if len(sys.argv) > 1:
    data = sys.argv[1]


with open(data) as fp:
    rag_results = RAGResults.from_json(fp.read())

# Start a persistent event loop in a separate thread
loop = asyncio.new_event_loop()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

t = Thread(target=start_loop, args=(loop,))
t.start()

def my_llm_api_func(prompts):
    """
    Get responses from LLM for the input prompts.
    """
    future = asyncio.run_coroutine_threadsafe(generate_all(prompts), loop)
    return future.result()

async def generate(prompt):
    """
    Generate a response for a single prompt.
    """
    try:
        resp = await client.generate(model='mistral-nemo', prompt=prompt)
        return resp["response"]
    except Exception as e:
        return f"Error: {e}"

async def generate_all(prompts, limit=100):
    """
    Generate responses for all prompts concurrently, with a limit on the number of concurrent tasks.
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def sem_generate(prompt):
        async with semaphore:
            return await generate(prompt)
    
    tasks = [sem_generate(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Set up the evaluator
evaluator = RAGChecker(
    custom_llm_api_func=my_llm_api_func
)

# Perform the evaluation
evaluator.evaluate(rag_results, all_metrics)

# After evaluation, stop and close the event loop
def stop_event_loop(loop):
    loop.call_soon_threadsafe(loop.stop)

# Stop the loop
stop_event_loop(loop)

# Wait for the thread to finish
t.join()

# Close the loop
loop.close()

# Output the results and elapsed time
print(rag_results)
print("--- %s seconds ---" % (time.time() - start_time))
