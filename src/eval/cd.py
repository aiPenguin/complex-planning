"""
Code borrowed from https://github.com/DreamLM/Dream/tree/main
"""
import re
from collections import Counter

from src.utils.eval_utils import read_jsonl, write_jsonl


EVAL_DIR = "data"


def cd_metric(inputs, preds):
    def check_eq(left_str, right_str):
        left_matches = re.match(r'(\d+)([+\-*/])(\d+)', left_str)
        if left_matches:
            return eval(left_str) == float(right_str)
        else:
            return False

    cor = 0

    for query, pred in zip(inputs, preds):
        subequations = pred.split(',')  # sub-equations
        match = True
        query_numbers = Counter(query.split(',')[:-1])
        for subeq in subequations:
            try:
                left, right = subeq.split('=')
                match &= check_eq(left, right)
                left_side_numbers = re.findall(r'(\d+)(?=[+-/*=])', subeq)
                query_numbers.subtract(left_side_numbers)
                query_numbers.update({right:1})
            except:
                match = False
            if not match:
                break

        answer = query.split(',')[-1]
        pred_ans = pred.split('=')[-1]

        query_numbers.subtract({query.split(',')[-1]: 1})
        numbers_match = all(value == 0 for value in query_numbers.values())
        # if not numbers_match:
        #     print(query +"\t" + label + "\t" + pred)
        cor += (match and (answer == pred_ans) and numbers_match)
        # cor += (match and (answer == pred_ans))

    return cor/len(preds)


def eval_cd3(generator, prediction_path=None):
    data = read_jsonl(EVAL_DIR+"/cd3_test.jsonl")
    n_few_shots = 8
    template = "Given 4 numbers, use +-*/ to operate over the first three numbers to achieve the last number.\n\n"
    template += "\n\n".join([f"Input: {i['input']}\nOutput: {i['output']}" for i in data[:n_few_shots]]) \
        + "\n\nInput: {input}\nOutput: "
    data = data[n_few_shots:]

    inputs = [template.format(input=i['input']) for i in data]
    print("Example input: ", inputs[0])
    generations = generator.generate(inputs)
    generations = [g.split('<|end_of_text|>')[0].split('\n')[0] for g in generations]
    print("Acc: ", cd_metric([i['input'] for i in data], generations))
    if prediction_path is not None:
        write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path)


def eval_cd4(generator, prediction_path=None):
    data = read_jsonl(EVAL_DIR+"/cd4_test.jsonl")
    n_few_shots = 8
    template = "Given 5 numbers, use +-*/ to operate over the first four numbers to achieve the fifth number.\n\n"
    template += "\n\n".join([f"Input: {i['input']}\nOutput: {i['output']}" for i in data[:n_few_shots]]) \
        + "\n\nInput: {input}\nOutput: "
    data = data[n_few_shots:]

    inputs = [template.format(input=i['input']) for i in data]
    print("Example input: ", inputs[0])
    generations = generator.generate(inputs)
    generations = [g.split('<|end_of_text|>')[0].split('\n')[0] for g in generations]
    print("Acc: ", cd_metric([i['input'] for i in data], generations))
    if prediction_path is not None:
        write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path)


def eval_cd5(generator, prediction_path=None):
    data = read_jsonl(EVAL_DIR+"/cd5_test.jsonl")
    n_few_shots = 8
    template = "Given 6 numbers, use +-*/ to operate over the first five numbers to achieve the last number.\n\n"
    template += "\n\n".join([f"Input: {i['input']}\nOutput: {i['output']}" for i in data[:n_few_shots]]) \
        + "\n\nInput: {input}\nOutput: "
    data = data[n_few_shots:]

    inputs = [template.format(input=i['input']) for i in data]
    print("Example input: ", inputs[0])
    generations = generator.generate(inputs)
    generations = [g.split('<|end_of_text|>')[0].split('\n')[0] for g in generations]
    print("Acc: ", cd_metric([i['input'] for i in data], generations))
    if prediction_path is not None:
        write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path)