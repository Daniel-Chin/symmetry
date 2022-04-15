from __future__ import annotations
from functools import lru_cache
import numpy as np

__all__ = ['Body', 'initTwoBody', 'stepTime']

BIG_G = 1
DT = .01

class Body:
    def __init__(self) -> None:
        self.radius = 1
        self.position = np.zeros((3, ))
        self.velocity = np.zeros((3, ))
    
    @lru_cache(2)
    def mass(self):
        return self.radius ** 3
    
    def stepTime(self, dt, total_force):
        self.velocity += dt * total_force / self.mass()
        self.position += dt * self.velocity
    
    def gravity(self, other: Body):
        displace = other.position - self.position
        norm_displace = np.linalg.norm(norm_displace)
        return (
            BIG_G * self.mass() * other.mass() * displace
        ) / norm_displace ** 3
    
    def randomize(self):
        self.position = np.random.normal(0, 1, 3)
        self.velocity = np.random.normal(0, 1, 3)

def initTwoBody():
    body_0 = Body()
    body_1 = Body()
    body_0.radius = 2
    body_1.radius = 1
    return body_0, body_1

def stepTime(time, body_0: Body, body_1: Body):
    while time > 0:
        if time < DT:
            dt = time
            time = -1   # make sure to exit loop
        else:
            dt = DT
            time -= dt
        force = body_0.gravity(body_1)
        body_0.stepTime(dt, + force)
        body_1.stepTime(dt, - force)
