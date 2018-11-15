import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import cv2


class gameOb():
    def __init__(self,coordinates, name, isReach=False):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.name = name
        self.isReach = isReach

class Fourrooms():
    def __init__(self, partial = None, hasObstacle=None, nTasks=2):
        self.sizeX = 11
        self.sizeY = 11
        self.actions = 4
        self.partial = partial
        self.hasObstacle = hasObstacle
        self.nTasks = nTasks

        self.walls = []
        for i in range(self.sizeX):
            self.walls.append((i, 0))
            self.walls.append((i, self.sizeX-1))
            self.walls.append((0, i))
            self.walls.append((self.sizeX-1, i))

            if (self.hasObstacle is not None):
                self.walls.append((i, 5))
                self.walls.append((5, i))

        if (self.hasObstacle is not None):
            self.walls.remove((0, 5))
            self.walls.remove((3, 5))
            self.walls.remove((7, 5))
            self.walls.remove((10, 5))

            self.walls.remove((5, 0))
            self.walls.remove((5, 3))
            self.walls.remove((5, 7))
            self.walls.remove((5, 10))

        self.walls.remove((0, 0))
        self.walls.remove((self.sizeX-1, self.sizeX-1))
        self.walls.remove((0, self.sizeX-1))
        self.walls.remove((self.sizeX-1, 0))

        self.initStartPoints()
        self.reset()

    def reset(self):
        self.agent = None
        self.goal1 = None
        self.goal2 = None
        self.goal3 = None
        self.fire1 = None
        self.fire2 = None

        self.agent = gameOb(self.newPosition(), 'agent')

        self.goal1 = gameOb(self.newPosition(), 'goal1')
        self.goal2 = gameOb(self.newPosition(), 'goal2')

        if (self.nTasks == 3):
            self.goal3 = gameOb(self.newPosition(), 'goal3')

        self.fire1 = gameOb(self.newPosition(), 'fire')
        self.fire2 = gameOb(self.newPosition(), 'fire')

        full_state, partial_state = self.renderEnv()
        return self.encode((self.agent.x, self.agent.y)), full_state, partial_state

    def samePosition(self, obj1, obj2):
        return obj1.x == obj2.x and obj1.y == obj2.y

    def isWall(self, position):
        for pos in self.walls:
            if (pos[0] == position[0] and pos[1] == position[1]):
                return True

        return False

    def initStartPoints(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        self.points = []
        for t in itertools.product(*iterables):
            self.points.append(t)

        # remove wall
        for pos in self.walls:
            if pos in self.points:
                self.points.remove(pos)

    def newPosition(self):
        points = self.points[:]
        currentPositions = []
        if (self.agent is not None):
            currentPositions.append((self.agent.x, self.agent.y))
        if (self.goal1 is not None):
            currentPositions.append((self.goal1.x, self.goal1.y))
        if (self.goal2 is not None):
            currentPositions.append((self.goal2.x, self.goal2.y))
        if (self.goal3 is not None):
            currentPositions.append((self.goal3.x, self.goal3.y))

        if (self.fire1 is not None):
            currentPositions.append((self.fire1.x, self.fire1.y))
        if (self.fire2 is not None):
            currentPositions.append((self.fire2.x, self.fire2.y))

        for pos in currentPositions:
            if pos in points:
                points.remove(pos)

        location = np.random.choice(range(len(points)))
        return points[location]

    def encode(self, state):
        if (state[0] == self.goal1.x and state[1] == self.goal1.y):
            return 0
        elif (state[0] == self.goal2.x and state[1] == self.goal2.y):
            return 1
        elif (self.goal3 is not None and state[0] == self.goal3.x and state[1] == self.goal3.y):
            return 2

        return -1

    def decode_goal(self, goal_idx):
        if (goal_idx == 0):
            goal = (self.goal1.x, self.goal1.y)
        if (goal_idx == 1):
            goal = (self.goal2.x, self.goal2.y)
        if (goal_idx == 2):
            goal = (self.goal3.x, self.goal3.y)

        return self.getImage(goal)

    def getImage(self, state):
        a = np.zeros([self.sizeX, self.sizeY, 3], dtype='uint8')
        a[:, :, :] = 128
        a[1:-1, 1:-1, :] = 255

        if (self.hasObstacle is not None):
            a[1:3, 5, :] = 128
            a[4:7, 5, :] = 128
            a[8:10, 5, :] = 128
            a[5, 1:3, :] = 128
            a[5, 4:7, :] = 128
            a[5, 8:10, :] = 128

        # goal1 is yellow
        if (self.goal1 is not None and self.goal1.isReach == False):
            a[self.goal1.x, self.goal1.y, 0] = 0
            a[self.goal1.x, self.goal1.y, 1] = 0
            a[self.goal1.x, self.goal1.y, 2] = 255

        # goal2 is green
        if (self.goal2 is not None and self.goal2.isReach == False):
            a[self.goal2.x, self.goal2.y, 0] = 0
            a[self.goal2.x, self.goal2.y, 1] = 255
            a[self.goal2.x, self.goal2.y, 2] = 0

        # goal3 is cyan
        if (self.goal3 is not None and self.goal3.isReach == False):
            a[self.goal3.x, self.goal3.y, 0] = 0
            a[self.goal3.x, self.goal3.y, 1] = 255
            a[self.goal3.x, self.goal3.y, 2] = 255

        # fire1 is red
        if (self.fire1 is not None):
            a[self.fire1.x, self.fire1.y, 0] = 255
            a[self.fire1.x, self.fire1.y, 1] = 0
            a[self.fire1.x, self.fire1.y, 2] = 0

        # goal1 is yellow
        if (self.fire2 is not None):
            a[self.fire2.x, self.fire2.y, 0] = 255
            a[self.fire2.x, self.fire2.y, 1] = 0
            a[self.fire2.x, self.fire2.y, 2] = 0

        # agent in black
        a[self.agent.x, self.agent.y, 0] = 0
        a[self.agent.x, self.agent.y, 1] = 0
        a[self.agent.x, self.agent.y, 2] = 0

        if self.partial is not None:
            if (self.partial == 3):
                a = a[state[0] - 1:state[0] + 2, state[1] - 1:state[1] + 2, :]
            else:
                partial_image = np.zeros([5, 5, 3], dtype='uint8')
                xmin = state[0] - 2
                xmax = state[0] + 3
                ymin = state[1] - 2
                ymax = state[1] + 3

                if (xmin < 0):
                    if (ymin < 0):
                        partial_image[1:5, 1:5, :] = a[0:4, 0:4, :]
                    elif (ymax > self.sizeY):
                        partial_image[1:5, 0:4, :] = a[0:4, self.sizeY - 4:self.sizeY, :]
                    else:
                        partial_image[1:5, 0:5, :] = a[0:4, ymin:ymax, :]
                elif (xmax > self.sizeX):
                    if (ymin < 0):
                        partial_image[0:4, 1:5, :] = a[self.sizeX - 4:self.sizeX, 0:4, :]
                    elif (ymax > self.sizeY):
                        partial_image[0:4, 0:4, :] = a[self.sizeX - 4:self.sizeX, self.sizeY - 4:self.sizeY, :]
                    else:
                        partial_image[0:4, 0:5, :] = a[self.sizeX - 4:self.sizeX, ymin:ymax, :]
                else:
                    if (ymin < 0):
                        partial_image[0:5, 1:5, :] = a[xmin:xmax, 0:4, :]
                    elif (ymax > self.sizeY):
                        partial_image[0:5, 0:4, :] = a[xmin:xmax, self.sizeY - 4:self.sizeY, :]
                    else:
                        partial_image[0:5, 0:5, :] = a[xmin:xmax, ymin:ymax, :]
                a = partial_image
        else:
            partial_image = np.zeros([self.sizeY, self.sizeY, 3], dtype='uint8')
            xmin = state[0] - 5
            xmax = state[0] + 6
            ymin = state[1] - 5
            ymax = state[1] + 6

            if (xmin < 0):
                if (ymin < 0):
                    partial_image[-xmin:self.sizeX, -ymin:self.sizeY, :] = a[0:self.sizeX + xmin, 0:self.sizeY + ymin,
                                                                           :]
                elif (ymax > self.sizeY):
                    partial_image[-xmin:self.sizeX, 0:self.sizeY + self.sizeY - ymax, :] = a[0:self.sizeX + xmin,
                                                                                           ymax - self.sizeY:self.sizeY,
                                                                                           :]
                else:
                    partial_image[-xmin:self.sizeX, 0:self.sizeY, :] = a[0:self.sizeX + xmin, ymin:ymax, :]
            elif (xmax > self.sizeX):
                if (ymin < 0):
                    partial_image[0:self.sizeX + self.sizeX - xmax, -ymin:self.sizeY, :] = a[
                                                                                           xmax - self.sizeX:self.sizeX,
                                                                                           0:self.sizeY + ymin, :]
                elif (ymax > self.sizeY):
                    partial_image[0:self.sizeX + self.sizeX - xmax, 0:self.sizeY + self.sizeY - ymax, :] = a[
                                                                                                         xmax - self.sizeX:self.sizeX,
                                                                                                         ymax - self.sizeY:self.sizeY,
                                                                                                         :]
                else:
                    partial_image[0:self.sizeX + self.sizeX - xmax, 0:self.sizeY, :] = a[xmax - self.sizeX:self.sizeX,
                                                                                       ymin:ymax, :]
            else:
                if (ymin < 0):
                    partial_image[0:self.sizeX, -ymin:self.sizeY, :] = a[xmin:xmax, 0:self.sizeY + ymin, :]
                elif (ymax > self.sizeY):
                    partial_image[0:self.sizeX, 0:self.sizeY + self.sizeY - ymax, :] = a[xmin:xmax,
                                                                                     ymax - self.sizeY:self.sizeY, :]
                else:
                    partial_image[0:self.sizeX, 0:self.sizeY, :] = a[xmin:xmax, ymin:ymax, :]

            a = partial_image

        patial = cv2.resize(a, (44, 44), interpolation=cv2.INTER_NEAREST)

        return patial

    def renderEnv(self):
        a = np.zeros([self.sizeX, self.sizeY, 3],dtype='uint8')
        a[:, :, :] = 128
        a[1:-1, 1:-1, :] = 255

        if (self.hasObstacle is not None):
            a[1:3, 5, :] = 128
            a[4:7, 5, :] = 128
            a[8:10, 5, :] = 128
            a[5, 1:3, :] = 128
            a[5, 4:7, :] = 128
            a[5, 8:10, :] = 128

        # goal1 is blue
        if (self.goal1 is not None and self.goal1.isReach == False):
            a[self.goal1.x, self.goal1.y, 0] = 0
            a[self.goal1.x, self.goal1.y, 1] = 0
            a[self.goal1.x, self.goal1.y, 2] = 255

        # goal2 is green
        if (self.goal2 is not None and self.goal2.isReach == False):
            a[self.goal2.x, self.goal2.y, 0] = 0
            a[self.goal2.x, self.goal2.y, 1] = 255
            a[self.goal2.x, self.goal2.y, 2] = 0

        # goal3 is cyan
        if (self.goal3 is not None and self.goal3.isReach == False):
            a[self.goal3.x, self.goal3.y, 0] = 0
            a[self.goal3.x, self.goal3.y, 1] = 255
            a[self.goal3.x, self.goal3.y, 2] = 255

        # fire1 is red
        if (self.fire1 is not None):
            a[self.fire1.x, self.fire1.y, 0] = 255
            a[self.fire1.x, self.fire1.y, 1] = 0
            a[self.fire1.x, self.fire1.y, 2] = 0

        # goal1 is yellow
        if (self.fire2 is not None):
            a[self.fire2.x, self.fire2.y, 0] = 255
            a[self.fire2.x, self.fire2.y, 1] = 0
            a[self.fire2.x, self.fire2.y, 2] = 0

        # agent in black
        a[self.agent.x, self.agent.y, 0] = 0
        a[self.agent.x, self.agent.y, 1] = 0
        a[self.agent.x, self.agent.y, 2] = 0

        total = cv2.resize(a, (44, 44), interpolation=cv2.INTER_NEAREST)

        # total[0, :, :] = 0
        # total[7, :, :] = 0
        # total[13, :, :] = 0
        # total[19, :, :] = 0
        # total[26, :, :] = 0
        # total[32, :, :] = 0
        # total[39, :, :] = 0
        # total[45, :, :] = 0
        # total[52, :, :] = 0
        # total[58, :, :] = 0
        # total[65, :, :] = 0
        # total[71, :, :] = 0
        # total[78, :, :] = 0
        # total[83, :, :] = 0
        #
        # total[:, 0, :] = 0
        # total[:, 7, :] = 0
        # total[:, 13, :] = 0
        # total[:, 19, :] = 0
        # total[:, 26, :] = 0
        # total[:, 32, :] = 0
        # total[:, 39, :] = 0
        # total[:, 45, :] = 0
        # total[:, 52, :] = 0
        # total[:, 58, :] = 0
        # total[:, 65, :] = 0
        # total[:, 71, :] = 0
        # total[:, 78, :] = 0
        # total[:, 83, :] = 0

        if self.partial is not None:
            if (self.partial == 3):
                a = a[self.agent.x - 1:self.agent.x + 2, self.agent.y - 1:self.agent.y + 2, :]
            else:
                partial_image = np.zeros([5, 5, 3], dtype='uint8')
                xmin = self.agent.x - 2
                xmax = self.agent.x + 3
                ymin = self.agent.y - 2
                ymax = self.agent.y + 3

                if (xmin < 0):
                    if (ymin < 0):
                        partial_image[1:5, 1:5, :] = a[0:4, 0:4, :]
                    elif (ymax > self.sizeY):
                        partial_image[1:5, 0:4, :] = a[0:4, self.sizeY - 4:self.sizeY, :]
                    else:
                        partial_image[1:5, 0:5, :] = a[0:4, ymin:ymax, :]
                elif (xmax > self.sizeX):
                    if (ymin < 0):
                        partial_image[0:4, 1:5, :] = a[self.sizeX - 4:self.sizeX, 0:4, :]
                    elif (ymax > self.sizeY):
                        partial_image[0:4, 0:4, :] = a[self.sizeX - 4:self.sizeX, self.sizeY - 4:self.sizeY, :]
                    else:
                        partial_image[0:4, 0:5, :] = a[self.sizeX - 4:self.sizeX, ymin:ymax, :]
                else:
                    if (ymin < 0):
                        partial_image[0:5, 1:5, :] = a[xmin:xmax, 0:4, :]
                    elif (ymax > self.sizeY):
                        partial_image[0:5, 0:4, :] = a[xmin:xmax, self.sizeY - 4:self.sizeY, :]
                    else:
                        partial_image[0:5, 0:5, :] = a[xmin:xmax, ymin:ymax, :]
                a = partial_image
        else:
            partial_image = np.zeros([self.sizeX, self.sizeY, 3], dtype='uint8')
            xmin = self.agent.x - 5
            xmax = self.agent.x + 6
            ymin = self.agent.y - 5
            ymax = self.agent.y + 6

            if (xmin < 0):
                if (ymin < 0):
                    partial_image[-xmin:self.sizeX, -ymin:self.sizeY, :] = a[0:self.sizeX+xmin, 0:self.sizeY+ymin, :]
                elif (ymax > self.sizeY):
                    partial_image[-xmin:self.sizeX, 0:self.sizeY+self.sizeY-ymax, :] = a[0:self.sizeX+xmin, ymax-self.sizeY:self.sizeY, :]
                else:
                    partial_image[-xmin:self.sizeX, 0:self.sizeY, :] = a[0:self.sizeX+xmin, ymin:ymax, :]
            elif (xmax > self.sizeX):
                if (ymin < 0):
                    partial_image[0:self.sizeX+self.sizeX-xmax, -ymin:self.sizeY, :] = a[xmax-self.sizeX:self.sizeX, 0:self.sizeY+ymin, :]
                elif (ymax > self.sizeY):
                    partial_image[0:self.sizeX+self.sizeX-xmax, 0:self.sizeY+self.sizeY-ymax, :] = a[xmax-self.sizeX:self.sizeX, ymax-self.sizeY:self.sizeY, :]
                else:
                    partial_image[0:self.sizeX+self.sizeX-xmax, 0:self.sizeY, :] = a[xmax-self.sizeX:self.sizeX, ymin:ymax, :]
            else:
                if (ymin < 0):
                    partial_image[0:self.sizeX, -ymin:self.sizeY, :] = a[xmin:xmax, 0:self.sizeY+ymin, :]
                elif (ymax > self.sizeY):
                    partial_image[0:self.sizeX, 0:self.sizeY+self.sizeY-ymax, :] = a[xmin:xmax, ymax-self.sizeY:self.sizeY, :]
                else:
                    partial_image[0:self.sizeX, 0:self.sizeY, :] = a[xmin:xmax, ymin:ymax, :]
            a = partial_image

        patial = cv2.resize(a, (44, 44), interpolation=cv2.INTER_NEAREST)

        return total, patial

    def getReward(self, subgoal_idx):
        done = 0.0
        reward = 0.0
        intrinsic_reward = 0.0
        extrinsic_reward = 0.0

        if (self.nTasks == 2):
            if (self.fire1.x == self.agent.x and self.fire1.y == self.agent.y):
                reward -= 1.0
                intrinsic_reward -= 1.0
                self.fire1 = gameOb(self.newPosition(), 'fire')
            elif (self.fire2.x == self.agent.x and self.fire2.y == self.agent.y):
                reward -= 1.0
                intrinsic_reward -= 1.0
                self.fire2 = gameOb(self.newPosition(), 'fire')
            elif (self.goal1.isReach == False and self.goal1.x == self.agent.x and self.goal1.y == self.agent.y):
                self.goal1.isReach = True
                if (self.goal2.isReach == True):
                    reward += 0.01
                    extrinsic_reward += 0.01
                    done = 1.0
                else:
                    reward += 1.0
                    extrinsic_reward += 1.0

                if (subgoal_idx == 0):
                    intrinsic_reward += 1.0

                    # #####
                    # done = 1.0
                    # #####
                else:
                    intrinsic_reward -= 1.0

            elif (self.goal2.isReach == False and self.goal2.x == self.agent.x and self.goal2.y == self.agent.y):
                self.goal2.isReach = True
                if (self.goal1.isReach == False):
                    reward += 0.01
                    extrinsic_reward += 0.01
                else:
                    reward += 1.0
                    extrinsic_reward += 1.0
                    done = 1.0

                if (subgoal_idx == 1):
                    intrinsic_reward += 1.0

                    # #####
                    # done = 1.0
                    # #####
                else:
                    intrinsic_reward -= 1.0
        else:
            if (self.fire1.x == self.agent.x and self.fire1.y == self.agent.y):
                reward -= 1.0
                intrinsic_reward -= 1.0
                self.fire1 = gameOb(self.newPosition(), 'fire')
            elif (self.fire2.x == self.agent.x and self.fire2.y == self.agent.y):
                reward -= 1.0
                intrinsic_reward -= 1.0
                self.fire2 = gameOb(self.newPosition(), 'fire')
            elif (self.goal1.isReach == False and self.goal1.x == self.agent.x and self.goal1.y == self.agent.y):
                self.goal1.isReach = True
                if (self.goal2.isReach == True or self.goal3.isReach == True):
                    reward += 0.01
                    extrinsic_reward += 0.01
                else:
                    reward += 1.0
                    extrinsic_reward += 1.0

                if (subgoal_idx == 0):
                    intrinsic_reward += 1.0
                else:
                    intrinsic_reward -= 1.0

            elif (self.goal2.isReach == False and self.goal2.x == self.agent.x and self.goal2.y == self.agent.y):
                self.goal2.isReach = True
                if (self.goal1.isReach == False or self.goal3.isReach == True):
                    reward += 0.01
                    extrinsic_reward += 0.01
                else:
                    reward += 1.0
                    extrinsic_reward += 1.0

                if (subgoal_idx == 1):
                    intrinsic_reward += 1.0
                else:
                    intrinsic_reward -= 1.0
            elif (self.goal3.isReach == False and self.goal3.x == self.agent.x and self.goal3.y == self.agent.y):
                self.goal3.isReach = True
                if (self.goal1.isReach == False or self.goal2.isReach == False):
                    reward += 0.01
                    extrinsic_reward += 0.01
                else:
                    reward += 1.0
                    extrinsic_reward += 1.0

                if (subgoal_idx == 2):
                    intrinsic_reward += 1.0
                else:
                    intrinsic_reward -= 1.0

            if (self.goal1.isReach == True and self.goal2.isReach == True and self.goal3.isReach == True):
                done = 1.0

        return intrinsic_reward, extrinsic_reward, reward, done

    def step(self, action, subgoal_idx):
        x = self.agent.x
        y = self.agent.y

        # 0 - up, 1 - down, 2 - left, 3 - right
        if action == 0:
            self.agent.y -= 1
        if action == 1:
            self.agent.y += 1
        if action == 2:
            self.agent.x -= 1
        if action == 3:
            self.agent.x += 1

        if (self.isWall((self.agent.x, self.agent.y))):
            self.agent.x = x
            self.agent.y = y

        state_idx = self.encode((self.agent.x,self.agent.y))

        intrinsic_reward, extrinsic_reward, reward, done = self.getReward(subgoal_idx)

        full_state, partial_state = self.renderEnv()
        return state_idx, full_state, partial_state, intrinsic_reward, extrinsic_reward, reward, done