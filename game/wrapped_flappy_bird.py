import random
from itertools import cycle
import numpy as np
import pygame

import game.flappy_bird_utils as flappy_bird_utils

FPS = 45
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        self.pipeVelX = -4
        self.playerVelY = 0
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -6
        self.playerFlapped = False
        self.playerRot = 45
        self.playerVelRot = 3
        self.playerRotThr = 20

    def frame_step(self, input_actions):
        pygame.event.pump()
        reward = 0
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        if input_actions[1] == 1:
            reward -= 0.12
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        next_pipe_idx = 0
        for i, pipe in enumerate(self.upperPipes):
            if pipe['x'] + PIPE_WIDTH > self.playerx:
                next_pipe_idx = i
                break

        upper_pipe = self.upperPipes[next_pipe_idx]
        lower_pipe = self.lowerPipes[next_pipe_idx]
        
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        pipeMidPos = upper_pipe['x'] + PIPE_WIDTH / 2
        isDirca = checkDirca({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex}, self.upperPipes, self.lowerPipes)
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)

        # ПРОХОЖДЕНИЕ ТРУБЫ
        d_reward_1 = 5
        # награда даётся небольшое время после прохождения трубы, а не один раз
        if pipeMidPos <= playerMidPos < pipeMidPos + playerMidPos:
            if not isCrash:
                reward += d_reward_1
                
        if pipeMidPos <= playerMidPos < pipeMidPos + 3:
            self.score += 1
            print("""\nGOL\n""")
                

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
            self.playerRot += 45
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        coords = get_info({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex}, self.upperPipes, self.lowerPipes)

        # ЛЕТИТ К ПРОЁМУ МЕЖДУ ТРУБАМИ (по оси y птичка находится между трубами)
        d_reward_2p = 2
        d_reward_2m = - 0.2
        if isDirca:
            reward += d_reward_2p
        else:
            reward += d_reward_2m

        # СТОЛКНОВЕНИЕ 
        d_reward_3 = - 9
        if isCrash:
            terminal = True
            reward += d_reward_3  # Большое наказание за столкновение
            self.score = 0
            self.__init__()

        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            visibleRot = self.playerRot
        playerSurface = pygame.transform.rotate(IMAGES['player'][self.playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (self.playerx, self.playery))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

        return coords, reward, terminal


def getRandomPipe():
    """returns a randomly generated pipe"""
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs) - 1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                                player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def checkDirca(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                                player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes[:1], lowerPipes[:1]):
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)


            if uPipeRect.y + uPipeRect.height <  playerRect.y+playerRect.height/2 < lPipeRect.y:
                return True

    return False

def get_info(player, upperPipes, lowerPipes):
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                                player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes[:1], lowerPipes[:1]):
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            return [playerRect.y+playerRect.height/2, uPipeRect.y + uPipeRect.height,  lPipeRect.y, uPipeRect.x - playerRect.x]
    

def pixelyCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    y1 = rect.y - rect1.y
    y2 = rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[-1][y1 + y] and hitmask2[-1][y2 + y]:
                return True
    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
