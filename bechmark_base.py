from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import asyncio
from ollama import AsyncClient
import time

start_time = time.time()
client = AsyncClient(host="prog3.student.famnit.upr.si:6666")

with open("data_to_bm.json") as fp:
    rag_results = RAGResults.from_json(fp.read())

def my_llm_api_func(prompts):
    """
    Get responses from LLM for the input prompts.
    Parameters
    ----------
    prompts: List[str]
        A list of prompts.
    
    Returns
    ----------
    response_list: List[str]
        A list of generated text.
    """
    resps = asyncio.run(generate_all(prompts))
    return resps

async def generate(prompt):
    """
    Generate a response for a single prompt.
    """
    
    # client = AsyncClient()
    resp =  await client.generate(model='mistral-nemo', prompt=prompt)
    return resp["response"]

async def generate_all(prompts):
    """
    Generate responses for all prompts concurrently.
    """
    tasks = [generate(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# set-up the evaluator
evaluator = RAGChecker(
    custom_llm_api_func=my_llm_api_func
)

# evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
evaluator.evaluate(rag_results, all_metrics)
print(rag_results)
print("--- %s seconds ---" % (time.time() - start_time))