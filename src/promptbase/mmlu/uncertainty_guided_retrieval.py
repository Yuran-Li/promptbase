import gzip
import json
import pathlib
from . import MMLU
from .mmlu_paths import mmlu_data_dir, mmlu_generations_dir

import math

def load_json_file(file_path):
    if type(file_path) is str:
        file_path = pathlib.Path(file_path)

    gz_path = file_path.with_suffix(file_path.suffix + ".gz")
    print(f"Looking for: {gz_path}")
    if gz_path.exists():
        print("Found zip file")
        with gzip.open(gz_path, "rt") as f:
            return json.load(f)
    else:
        print("Found regular file")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

def RAG_CoT_dev(dataset_name: str):
    dev_problem = f"mmlu_{dataset_name}_val"
    
    print(f"Starting check Shanon Entropy of {dataset_name} results...")
    p_correct = []
    list_retrieval = []
    dev_data = load_json_file(
            mmlu_generations_dir / "expt" / dev_problem / "cot" / "result.json"
        )
    for index, item in enumerate(dev_data):
            correct_n = 0
            correct_answer = item["correct_answer"]
            expt = item["expt"]
            total_n = len(expt)
            for key, value in expt.items():
                if value["answer"] == correct_answer:
                    correct_n += 1
            probability = correct_n/total_n
            p_correct.append(probability)
            if probability<0.9:
                list_retrieval.append({"question_number": item["question_number"], 
                                       "question": item["question"],
                                       "correct_answer": item["correct_answer"],
                                       "has_media": item["has_media"],
                                       "dataset": item["dataset"],
                                       "id": item["id"],
                                       "split": item["split"],
                                       "extra": item["extra"],
                                       "answer_choices": item["answer_choices"]
                                       })

    MMLU.generate_solutions_step_by_step(
                    dev_problem, 
                    run_name=f"{dev_problem}/rag_cot_step_by_step", 
                    rag_problems = list_retrieval, 
                    model = "gpt-4o-mini")
            
def RAG_CoT_test(dataset_name: str):
    test_problem = f"mmlu_{dataset_name}_test"
    
    print(f"Starting check Shanon Entropy of {dataset_name} results...")
    list_retrieval = []
    dev_data = load_json_file(
            mmlu_generations_dir / "expt" / test_problem / "cot" / "result.json"
        )
    for index, item in enumerate(dev_data):
            expt = item["expt"]
            answer_choices = item["answer_choices"]
            frequencies = [sum(1 for _, value in expt.items() if value["answer"] == answer)
                        for answer in answer_choices]

            entropy = shannon_entropy(frequencies)
            N = len(frequencies)
            if entropy > 0.8 * math.log2(N):
                list_retrieval.append({"question_number": item["question_number"], 
                                       "question": item["question"],
                                       "correct_answer": item["correct_answer"],
                                       "has_media": item["has_media"],
                                       "dataset": item["dataset"],
                                       "id": item["id"],
                                       "split": item["split"],
                                       "extra": item["extra"],
                                       "answer_choices": item["answer_choices"]
                                       })

    MMLU.generate_solutions_step_by_step(
                    test_problem, 
                    run_name=f"{test_problem}/rag_cot_step_by_step", 
                    rag_problems = list_retrieval, 
                    model = "gpt-4o-mini")

def shannon_entropy(frequencies):
    total_responses = sum(frequencies)
    if total_responses == 0:
        return 0  
    
    entropy = 0
    for freq in frequencies:
        if freq > 0:
            p_i = freq / total_responses  
            entropy -= p_i * math.log(p_i, 2)  
    return entropy