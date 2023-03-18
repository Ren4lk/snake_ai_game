import random
from utils import Moves

MAX_ITERATIONS = 100


class Snake():
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.score = 0
        self.apple = []
        self.head = [random.randint(1, width - 1),
                     random.randint(1, height - 1)]
        self.last_move = None
        self.__body = []
        self.__length = 1
        self.__generateApple()
        self.__n_iterations = 0

    def reset(self):
        self.score = 0
        self.apple = []
        self.head = [random.randint(1, self.width - 1),
                     random.randint(1, self.height - 1)]
        self.last_move = None
        self.__body = []
        self.__length = 1
        self.__generateApple()
        self.__n_iterations = 0

    def isDanger(self, point: list[int, int]):
        if point[0] >= self.width or point[0] < 0 or point[1] >= self.height or point[1] < 0:
            return True
        if self.head == point:
            return True
        for coord in self.__body[:-1]:
            if coord == point:
                return True
        return False

    def getSnake(self) -> list[list[int, int]]:
        snake = []
        snake.append(self.head)
        for x in self.__body:
            snake.append(x)
        return snake

    def move(self, action: Moves):
        if self.__length > 1 and (self.last_move == Moves.UP and action == Moves.DOWN or
                                  self.last_move == Moves.DOWN and action == Moves.UP or
                                  self.last_move == Moves.LEFT and action == Moves.RIGHT or
                                  self.last_move == Moves.RIGHT and action == Moves.LEFT):
            action = self.last_move

        reward = 0
        game_over = False
        if action != None:
            self.__n_iterations += 1
            if action == Moves.UP:
                self.__move(0, -1)
            elif action == Moves.DOWN:
                self.__move(0, 1)
            elif action == Moves.LEFT:
                self.__move(-1, 0)
            elif action == Moves.RIGHT:
                self.__move(1, 0)
            self.last_move = action

            if self.__isBordersReached() or self.__isHeadInBody():
                game_over = True
                reward = -10
            else:
                if self.__eatApple():
                    self.score += 1
                    self.__addCell()
                    self.__generateApple()
                    reward = 10
                    self.__n_iterations = 0
                else:
                    if self.__n_iterations >= MAX_ITERATIONS:
                        reward = -10
        return reward, game_over

    def __generateApple(self):
        coordinates = []
        for x in range(self.width - 1):
            for y in range(self.height - 1):
                coordinates.append([x, y])
        coordinates = [x for x in coordinates if x not in self.getSnake()]
        self.apple = random.choice(coordinates)

    def __addCell(self):
        x = 0
        y = 0
        if self.__length == 1:
            if self.last_move == Moves.UP:
                x = self.head[0]
                y = self.head[1] - 1
            elif self.last_move == Moves.DOWN:
                x = self.head[0]
                y = self.head[1] + 1
            elif self.last_move == Moves.LEFT:
                x = self.head[0] - 1
                y = self.head[1]
            elif self.last_move == Moves.RIGHT:
                x = self.head[0] + 1
                y = self.head[1]
        else:
            x = self.__body[-1][0]
            y = self.__body[-1][1]
        self.__body.append([x, y])
        self.__length += 1

    def __eatApple(self) -> bool:
        if self.apple == self.head:
            del self.apple
            return True
        else:
            return False

    def __isBordersReached(self):
        if self.head[0] >= self.width or self.head[0] < 0 or self.head[1] >= self.height or self.head[1] < 0:
            return True
        else:
            return False

    def __isHeadInBody(self):
        result = False
        for x in self.__body:
            if x == self.head:
                result = True
                break
        return result

    def __move(self, x, y):
        head = [self.head[0] + x, self.head[1] + y]
        body = []
        body.append(self.head)
        for coordinate in self.__body:
            body.append(coordinate)
        del body[-1]
        self.__body = body
        self.head = head

    def __len__(self):
        return self.__length
