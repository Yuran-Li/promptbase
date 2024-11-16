import concurrent
import concurrent.futures
import datetime
import json
import logging
import os
import random
import re
import string
import time
import types

import numpy as np
import requests
from tqdm import tqdm

import openai

logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARN"))

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("warnings.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


################################################################################
# File-Related
################################################################################
def load_jsonl(file_name):
    # Load a jsonl file as list of objects
    with open(file_name) as f:
        lines = f.readlines()

    return [json.loads(line) for line in lines if line]


def dump_jsonl(objs, file_name):
    # Save a list/dict of objects to jsonl
    # If it is dict, we only save the values

    with open(file_name, "w") as f:
        if type(objs) is dict:
            for key in objs:
                f.write(json.dumps(objs[key]) + "\n")
        else:
            for obj in objs:
                f.write(json.dumps(obj) + "\n")


################################################################################
# String-Related
################################################################################


def now_string():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def random_string(length=6):
    return "".join(random.choices(string.digits + string.ascii_letters, k=length))


################################################################################
# Multithreading
################################################################################


def run_batch_jobs(run_task, tasks, max_thread):
    """
    Run a batch of tasks with cache.
    - run_task: the function to be called
    - tasks: the list of input for the function
    - max_thread: the number of thread we use
    """
    results = []
    max_failures = 10
    observed_failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_thread) as executor, tqdm(
        total=len(tasks)
    ) as pbar:
        futures = [executor.submit(run_task, task) for task in tasks]

        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.exception("Error occurred during run_batch_jobs.")
                observed_failures += 1
                if observed_failures > max_failures:
                    raise

    return results


################################################################################
# OpenAI API Tools
################################################################################
openai.api_key = f"{os.getenv('OPENAI_API_KEY')}"

def embed(text, model_name="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(
                input=text,
                model=model_name
            )
        data = response["data"]
        
        if isinstance(text, str):
            return data[0]["embedding"]
        else:
            sorted_data = sorted(data, key=lambda x:x["index"])
            return [item["embedding"] for item in sorted_data]
    except openai.OpenAIError as e:
        print(f"Error while generating embedding: {e}")
        exit()   


def embed_batch(texts, model_name="text-embedding-ada-002", batch_size=100):
    embeddings = []
    # Split the input list into sublists of size batch_size
    batches = np.array_split(
        texts,
        len(texts) // batch_size
        if len(texts) % batch_size == 0
        else len(texts) // batch_size + 1,
    )
    for batch in tqdm(batches):
        batch = list(
            batch
        )  # Convert batch back to list if it isn't (depends on array_split output)
        batch_embeddings = embed(batch, model_name)
        embeddings.extend(batch_embeddings)
    return embeddings


def text_completion_impl(
    prompt,
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=3500,
    top_p=1.0,
    logprobs=10,
    stop="<|diff_marker|>",
    echo=False,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    max_trial=100,
    **kwargs,
):
    """
    Performs text completion using the openai API with
    - prompt (str or array of str)
    - model ("text-davinci-003", "text-davinci-002", ...)
    - tempature (0 for picking the best token, 1 for more creative solution)
    - max_tokens (limit the total number of generated tokens. 8193 is the maximum context length text-alpha-002)
    - max_trial (the number of retry after getting rate limited warning, we rethrow for all other errors)
    - logprobs (return a list of the most likely tokens and its probabilites. either integer in [1,5] or None)
    - stop (string or list of string (up to 4 strings). The returned text will not contain the stop sequence.)
    """
    for attempt in range(max_trial):
        try:
            # 控制重试间隔
            time.sleep(random.uniform(0.1, 0.5))

            if isinstance(prompt, list) and isinstance(prompt[0], dict):
                # Chat-style prompt
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop=stop,
                )
            else:
                # Completion-style prompt
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=logprobs,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop=stop,
                )

            # 返回结果
            return {
                "response": response,
                "text": response.choices[0].get("text", "") if "text" in response.choices[0] else response.choices[0]["message"]["content"],
                "success": True,
            }

        except openai.OpenAIError as e:
            logging.error(f"Attempt {attempt + 1}/{max_trial} failed: {str(e)}")
            if attempt + 1 == max_trial:
                return {"error": str(e), "success": False}

    return  {"error": "Max attempts exceeded", "success": False}

def text_completion(**kwargs):
    result = text_completion_impl(**kwargs)
    if "log_file" in kwargs:
        message = "########## Prompt ##########\n"
        message += (
            str(kwargs["prompt"])
            + "\nmax_tokens="
            + str(kwargs.get("max_tokens", 0))
            + "\n"
        )
        message += "########## Response ##########\n"
        message += result.get("text", "NONE") + "\n"
        with open(kwargs["log_file"], "a") as f:
            f.write(message)
    return result
