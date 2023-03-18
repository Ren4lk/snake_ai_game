import os
import torch
import random
import numpy as np
from collections import deque
import utils
from model import Linear_QNet, QTrainer
from SimpleSnake import Snake, Moves


class Agent:
    def __init__(self, hidden_size: int):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=utils.MAX_MEMORY)
        self.hidden_size = hidden_size
        self.model = Linear_QNet(
            utils.IN_SIZE, self.hidden_size, utils.OUT_SIZE)
        self.trainer = QTrainer(self.model, lr=utils.LR, gamma=self.gamma)

    def get_state(self, snake: Snake):
        head = snake.head
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]

        dir_u = snake.last_move == Moves.UP
        dir_d = snake.last_move == Moves.DOWN
        dir_l = snake.last_move == Moves.LEFT
        dir_r = snake.last_move == Moves.RIGHT

        state = [
            # Danger straight
            (dir_r and snake.isDanger(point_r)) or
            (dir_l and snake.isDanger(point_l)) or
            (dir_u and snake.isDanger(point_u)) or
            (dir_d and snake.isDanger(point_d)),
            # Danger right
            (dir_u and snake.isDanger(point_r)) or
            (dir_d and snake.isDanger(point_l)) or
            (dir_l and snake.isDanger(point_u)) or
            (dir_r and snake.isDanger(point_d)),
            # Danger left
            (dir_d and snake.isDanger(point_r)) or
            (dir_u and snake.isDanger(point_l)) or
            (dir_r and snake.isDanger(point_u)) or
            (dir_l and snake.isDanger(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,
            snake.apple[0] < snake.head[0],
            snake.apple[0] > snake.head[0],
            snake.apple[1] < snake.head[1],
            snake.apple[1] > snake.head[1]
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > utils.BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, utils.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        move = 0
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(hidden_layer_size: int, weights_path=None):
    record = 0
    agent = Agent(hidden_layer_size)
    if weights_path != None:
        agent.model.load_state_dict(torch.load(weights_path))
    game = Snake(int(utils.WIDTH/utils.BLOCK_SIZE),
                 int(utils.HEIGHT/utils.BLOCK_SIZE))
    while True:
        old_state = agent.get_state(game)

        final_move = agent.get_action(old_state)

        reward, done = game.move(utils.convertTensorToMove(final_move))
        score = game.score

        state_new = agent.get_state(game)

        agent.train_short_memory(
            old_state, final_move, reward, state_new, done)

        agent.remember(old_state, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            if agent.n_games == utils.N_EPOCHS:
                torch.save(agent.model.state_dict(), os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), f'model_game_hidden_layers_{hidden_layer_size}-record-{record}.pth'))
                break

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    for n in utils.HIDDEN_SIZES:
        train(n)
