# Simple_simulator guiObjects
# Wan Ju Kang
# Dec. 1, 2017

# ----------------------------------------------------------------
# Purpose of Simple Simulator guiObjects is to provide abstraction
# for the drawn target and drones on the Simple Simulator Canvas.
# To use guiObjects, make an instance and blit it on a surface.
# ----------------------------------------------------------------

import pygame
from math import cos, sin, pi
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 100, 0, 128)
BLUE = (0, 128, 255, 128)
RED = (255, 0, 0, 128)

class guiPrey():
    def __init__(self, xi = 1, yi = 1, zi = 30, ai = 10, prey_id = 0):
        # Take initial (x, y, z, radius) as argument
        self.x = xi
        self.y = yi
        self.z = zi
        # self.tr = tr
        self.a = ai
        self.a = None # Not used
        self.prey_id = prey_id
        self.name = "prey"

        # Label the prey
        self.color = ORANGE
        self.fs = 15 # font size
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), self.fs)
        self.text = ""
        self.label = self.font.render(self.text, True, BLACK)
        
    def setup(self, sx = 25, sy = 25):
        # Set up prey's surface
        self.sx = sx
        self.sy = sy
        self.surface = pygame.Surface((self.sx, self.sy), pygame.SRCALPHA)
        self.surface = self.surface.convert_alpha()

class guiPred():
    def __init__(self, xi = 1, yi = 1, zi = 30, ai = 0, pred_id = 0):
        # Take initial (x, y, z, yaw, drone_id) as argument
        self.x = xi
        self.y = yi
        self.z = zi
        self.a = ai
        self.pred_id = pred_id
        self.name = "predator" + str(pred_id)

        # Label the drone
        self.body_color = BLUE
        self.eye_color = RED
        self.fs = 32 # font size
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), self.fs)
        self.text = str(pred_id)
        self.label = self.font.render(self.text, True, BLACK)

        # Misc.
        self.eye_size = 0.4

    def setup(self, sx = 25, sy = 25):
        # Set up predator's surface
        self.sx = sx
        self.sy = sy
        self.surface = pygame.Surface((self.sx, self.sy), pygame.SRCALPHA)
        self.surface = self.surface.convert_alpha()
