import numpy as np
from sudoku import Sudoku
from time import time
f = open("sudoku_seeds.txt", 'r')
text = f.read().split('\n')
puzzle = Sudoku(3).difficulty(0.1)
n_sudokus_to_save = len(text)
sudoku_dict = dict()

a0 = time()
for line in range(4000):
    a = time()
    unsolved = text[line]
    sudoku_board = list(map(int, list(unsolved)))
    sudoku_board = [x if x > 0 else None for x in sudoku_board]
    sudoku_board = np.reshape(sudoku_board, (9, 9))
    puzzle.board = sudoku_board
    puzzle = puzzle.solve()
    sudoku_dict[line] = (sudoku_board, puzzle.board)
    print(time()-a)
print(time() - a0)
print(sudoku_dict[2])