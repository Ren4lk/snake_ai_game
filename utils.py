import torch
import enum

WHITE = (255, 255, 255)
YELLOW = (255, 255, 102)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (43, 43, 255)
APPLE_COLORS = [RED, GREEN, YELLOW]
SNAKE_COLORS = [(x, x, x) for x in range(0, 200, 20)]

WIDTH = 1200
HEIGHT = 1200

BLOCK_SIZE = 40
SPEED = 50

MAX_MEMORY = 1000000
BATCH_SIZE = 10000
LR = 0.0005

N_EPOCHS = 2000

IN_SIZE = 11
HIDDEN_SIZES = [32, 64, 128, 256, 512, 1024]
OUT_SIZE = 4

class Moves(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def convertTensorToMove(move: torch.tensor) -> Moves:
    number = 0
    for i in range(len(move) - 1):
        if move[i] == 1:
            number = i+1
            break
    if number == Moves.UP.value:
        return Moves.UP
    elif number == Moves.DOWN.value:
        return Moves.DOWN
    elif number == Moves.LEFT.value:
        return Moves.LEFT
    else:
        return Moves.RIGHT
    