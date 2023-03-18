import os
import torch
import pygame
import utils
from SimpleSnake import Snake, Moves
from agent import Agent

pygame.init()
display = pygame.display.set_mode((utils.WIDTH, utils.HEIGHT))
pygame.display.set_caption('Simple snake')
score_font = pygame.font.SysFont("comicsansms", 135)
clock = pygame.time.Clock()


def drawScore(score: int):
    value = score_font.render("Score: " + str(score), True, utils.BLUE)
    display.blit(value, [0, 0])


def drawBlocks(size: int, color: list, blocks: list):
    i = 0
    for block in blocks:
        pygame.draw.rect(display, color[i % len(color)], [
                         block[0]*size, block[1]*size, size, size])
        i += 1


def humanGameLoop(snake: Snake):
    game_over = False
    active = None
    game_exit = False
    while not game_exit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_exit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    active = Moves.UP
                elif event.key == pygame.K_DOWN:
                    active = Moves.DOWN
                elif event.key == pygame.K_LEFT:
                    active = Moves.LEFT
                elif event.key == pygame.K_RIGHT:
                    active = Moves.RIGHT

        reward, game_over = snake.move(active)

        display.fill(utils.WHITE)

        drawScore(snake.score)
        drawBlocks(utils.BLOCK_SIZE, utils.APPLE_COLORS, [snake.apple])
        drawBlocks(utils.BLOCK_SIZE, utils.SNAKE_COLORS, snake.getSnake())

        pygame.display.update()

        if game_over == True:
            snake.reset()

        clock.tick(utils.SPEED)

    pygame.quit()
    quit()


def aiGameLoop(snake: Snake, hidden_size: int, weights_path: str):
    game_over = False
    game_exit = False
    agent = Agent(hidden_size)
    agent.model.load_state_dict(torch.load(
        weights_path, map_location=torch.device('cuda')))
    while not game_exit:
        state = torch.tensor(agent.get_state(snake), dtype=torch.float)
        active = [0, 0, 0, 0]
        prediction = agent.model(state)
        move = torch.argmax(prediction).item()
        active[move] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_exit = True

        reward, game_over = snake.move(utils.convertTensorToMove(active))

        display.fill(utils.WHITE)

        drawScore(snake.score)
        drawBlocks(utils.BLOCK_SIZE, utils.APPLE_COLORS, [snake.apple])
        drawBlocks(utils.BLOCK_SIZE, utils.SNAKE_COLORS, snake.getSnake())

        pygame.display.update()

        if game_over == True:
            snake.reset()

        clock.tick(utils.SPEED)

    pygame.quit()
    quit()


if __name__ == '__main__':
    width = int(utils.WIDTH/utils.BLOCK_SIZE)
    height = int(utils.HEIGHT/utils.BLOCK_SIZE)
    snake = Snake(width, height)

    # humanGameLoop(snake)
    aiGameLoop(snake, 256, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'model_game_hidden_layers_256-record-100.pth'))
