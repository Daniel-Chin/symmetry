import math
import random

GRAVITY = 10.0
BOUNCE_ENERGY_LOSS = 0.19

ACCURACY_HEIGHT = 3
ACCURACY_TIME = 4
ACCURACY_VELOCITY = 3


def uniformLinerCalc(s, v, dt):
    return s + v * dt, v


def highestPosition(h, v, g=GRAVITY):
    time = v / g
    height = h + 0.5 * math.pow(v, 2.0) / g
    return round(height, ACCURACY_HEIGHT), 0, round(time, ACCURACY_TIME)


def fastestVelocity(h, v, g=GRAVITY):
    velocity = math.sqrt(2 * g * h + math.pow(v, 2.0))
    time = (velocity + v) / g
    return 0, round(velocity, ACCURACY_VELOCITY), round(time, ACCURACY_TIME)


def velocityAfterEnergyLoss(v):
    return round(v * math.sqrt(1-BOUNCE_ENERGY_LOSS), ACCURACY_VELOCITY)


def distanceCalc(v, dt, g=GRAVITY):
    return (v + v - dt * g) * dt / 2


def verticalCalc(h, v, dt, g=GRAVITY):
    if (h == 0 and v == 0): return 0.0, 0.0
    timeRemain = dt
    height = h
    velocity = v
    isUp = velocity > 0
    iter = 0
    while True:
        if isUp:
            nextHeight, nextVelocity, timeCost = highestPosition(height, velocity, g)
            isUp = False
        else:
            nextHeight, nextVelocity, timeCost = fastestVelocity(height, velocity, g)
            nextVelocity = velocityAfterEnergyLoss(nextVelocity)
            isUp = True
        iter += 1
        # print(iter, nextHeight, nextVelocity, timeCost, isUp)
        if timeRemain - timeCost <= 0:
            dh = distanceCalc(velocity, timeRemain, g)
            return height + dh, velocity - timeRemain * g
        else:
            timeRemain -= timeCost
            height = nextHeight
            velocity = nextVelocity


def ballNextState(sXYZ, vXYZ, dt, g=GRAVITY):
    nsX, nvX = uniformLinerCalc(sXYZ[0], vXYZ[0], dt)
    nsY, nvY = verticalCalc(sXYZ[1], vXYZ[1], dt, g)
    nsZ, nvZ = uniformLinerCalc(sXYZ[2], vXYZ[2], dt)
    return [nsX, nsY, nsZ], [nvX, nvY, nvZ]


def randomBallInitVelocity():
    return [random.random() * 4 - 2, random.random() * 20 - 10, random.random() * 4 - 2]


def gen_a_fix_len_traj(maxSeqLen, DT, init_ball_position, init_ball_velocity, g=GRAVITY):
    traj = []
    nextPosition = init_ball_position
    nextVelocity = init_ball_velocity
    for i in range(0, maxSeqLen):
        traj.append(nextPosition)
        nextPosition, nextVelocity = ballNextState(nextPosition, nextVelocity, DT, g)
    return traj



if __name__ == "__main__":
    init_ball_position = [0, 1, 0]
    init_ball_velocity = [0, 6, 0]
    traj_1 = gen_a_fix_len_traj(30, 0.2, init_ball_position, init_ball_velocity, 10)
    print([round(p[1], 2) for p in traj_1])
    init_ball_position_2 = [0, 2, 0]
    init_ball_velocity_2 = [0, 12, 0]
    traj_2 = gen_a_fix_len_traj(30, 0.2, init_ball_position_2, init_ball_velocity_2, 20)
    print([round(p[1]/2, 2) for p in traj_2])