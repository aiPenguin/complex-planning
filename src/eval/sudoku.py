"""
Code borrowed from https://github.com/DreamLM/Dream/tree/main
"""
import numpy as np

from src.utils.eval_utils import read_jsonl, write_jsonl


EVAL_DIR = "data"


def is_valid_sudoku(input, prediction):
    ### check 4x4 sudoku result

    prediction = prediction[:len(input)]
    input_array = np.array([list(map(int, row)) for row in input.strip().split('\n')])
    try:
        grid = np.array([list(map(int, row)) for row in prediction.strip().split('\n')])
        if grid.shape != (4,4):
            return False
    except:
        return False
    
    # Create a mask for non-zero positions in the input
    non_zero_mask = input_array != 0
    
    # Check if the non-zero positions in the input match the output
    if not np.all(input_array[non_zero_mask] == grid[non_zero_mask]):
        return False

    # Check if each row, column, and subgrid contains the digits 1 to 4
    expected_set = {1, 2, 3, 4}
    
    # Check rows
    for row in grid:
        if set(row) != expected_set:
            return False
    
    # Check columns
    for col in range(4):
        if set(grid[row][col] for row in range(4)) != expected_set:
            return False
    
    # Check 2x2 subgrids
    for start_row in (0, 2):
        for start_col in (0, 2):
            subgrid = {grid[r][c] for r in range(start_row, start_row + 2) for c in range(start_col, start_col + 2)}
            if subgrid != expected_set:
                return False

    return True

def eval_sudoku(generator, prediction_path=None):
    # for n in [4,5,6,7,8,9,10,11,12]:
    for n in [10]:
        data = read_jsonl(EVAL_DIR+f"/sudoku_4x4_{n}.jsonl")
        n_few_shots = 8
        template = "Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so that each column, each row, and each of the four 2x2 subgrids that compose the grid contains all of the digits from 1 to 4.\n\n"
        template += "\n\n".join([f"Input:\n{i['input']}\nOutput:\n{i['output']}" for i in data[:n_few_shots]]) \
            + "\n\nInput:\n{input}\nOutput:\n "
        data = data[n_few_shots:]

        inputs = [template.format(input=i['input']) for i in data]
        print("Example input: ", inputs[0])
        generations = generator.generate(inputs)
        generations = [g.split('<|endoftext|>')[0].split('\n\n')[0].replace(' ', '') for g in generations]
        # generations = [i["prediction"] for i in read_jsonl(prediction_path+f"_{n}")]
        acc = sum([is_valid_sudoku(i['input'], j) for i, j in zip(data, generations)]) / len(data)
        print(f"Filled n={n}, Acc: ", acc)
        if prediction_path is not None:
            write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path+f"_{n}")

if __name__ == "__main__":
    print(is_valid_sudoku("1000\n0002\n4003\n0000", "1234\n3412\n4123\n2341"))
    print(is_valid_sudoku("0020\n0034\n0400\n1000", "2413\n3142\n4321\n1234"))

