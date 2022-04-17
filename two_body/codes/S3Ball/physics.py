from __future__ import annotations
from functools import lru_cache
from typing import List
import numpy as np

__all__ = [
    'Body', 'initLegalTwoBody', 'stepTime', 'printTotal', 
    'distanceBetween', 'totalEnergy', 'oneLegalRun', 
]

POSITION_STD = 3
VELOCITY_STD = 1
BIG_G = 10
FINE_DT = .01   # very fine dt. This is for sim, not for DL. 

class Body:
    def __init__(self) -> None:
        self.radius = 1
        self.position = np.zeros((3, ))
        self.velocity = np.zeros((3, ))
    
    def snapshot(self):
        body = Body()
        body.radius = self.radius
        body.position = self.position.copy()
        body.velocity = self.velocity.copy()
        return body
    
    def __repr__(self):
        return f'<ball {self.position}>'
    
    @lru_cache(2)
    def mass(self):
        return self.radius ** 3
    
    def stepTime(self, dt, total_force):
        self.velocity += dt * total_force / self.mass()
        self.position += dt * self.velocity
    
    def gravity(self, other: Body):
        displace = other.position - self.position
        norm_displace = np.linalg.norm(displace)
        if norm_displace == 0:
            return np.zeros((3, ))
        else:
            return (
                BIG_G * self.mass() * other.mass() * displace
            ) / norm_displace ** 3

def initTwoBody():
    bodies = [Body(), Body()]
    # bodies[0].radius = 2
    bodies[0].radius = 1
    bodies[1].radius = 1
    d_pos = np.random.normal(0, POSITION_STD, 3)
    d_vel = np.random.normal(0, VELOCITY_STD, 3)
    total_mass = bodies[0].mass() + bodies[1].mass()
    weights = []
    for body in bodies:
        weights.append(body.mass() / total_mass)
    weights = [weights[1], - weights[0]]
    for body, weight in zip(bodies, weights):
        body.position = d_pos * weight
        body.velocity = d_vel * weight
    return bodies

def initLegalTwoBody():
    # rejection sampling
    while True:
        bodies = initTwoBody()
        # reject nothing
        return bodies    

def stepTime(time, body_0: Body, body_1: Body):
    while time > 0:
        if time < FINE_DT:
            dt = time
            time = -1   # make sure to exit loop
        else:
            dt = FINE_DT
            time -= dt
        force = body_0.gravity(body_1)
        body_0.stepTime(dt, + force)
        body_1.stepTime(dt, - force)

def printTotal(bodies: List[Body]):
    total_pos = np.zeros((3, ))
    total_mom = np.zeros((3, ))
    for body in bodies:
        total_pos += body.mass() * body.position
        total_mom += body.mass() * body.velocity
    print('质心, 总动量：', [total_pos, total_mom])

def distanceBetween(bodies: List[Body]):
    return np.linalg.norm(
        bodies[0].position - bodies[1].position
    )

def totalEnergy(bodies: List[Body]):
    energy = 0
    for body in bodies:
        energy += 0.5 * body.mass() * np.linalg.norm(
            body.velocity
        ) ** 2
    energy -= BIG_G * (
        bodies[0].mass() * bodies[1].mass()
    ) / distanceBetween(bodies)
    return energy

def oneRun(dt, n_frames):
    bodies = initLegalTwoBody()
    trajectory = []
    for _ in range(n_frames):
        trajectory.append([x.snapshot() for x in bodies])
        stepTime(dt, *bodies)
        verify(*bodies)
    return trajectory

class RejectThisSampleException(Exception): pass

def verify(a: Body, b: Body):
    for body in (a, b):
        if np.linalg.norm(body.position) > 8:
            raise RejectThisSampleException
    if np.linalg.norm(
        a.position - b.position
    ) < a.radius + b.radius:
        raise RejectThisSampleException

def oneLegalRun(*a, **kw):
    # rejection sampling
    while True:
        try:
            trajectory = oneRun(*a, **kw)
        except RejectThisSampleException:
            # print('Reject')
            continue
        else:
            # print('Accept')
            return trajectory

if __name__ == '__main__':
    print(*oneLegalRun(1, 30), sep='\n')
