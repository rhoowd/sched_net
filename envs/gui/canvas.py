# Simple_simulator Canvas
# Wan Ju Kang
# May 9, 2018

# ---------------------------------------------------------------
# Display on GUI the positions of predator agents and prey agents
# ---------------------------------------------------------------

import random
import socket
import threading
import json
import pygame
import ConfigParser

from time import sleep
from math import pi, sin, cos, sqrt, ceil, floor
from envs.gui.guiObjects import guiPred, guiPrey # no need for camera view

GREY = (25, 25, 25, 128)
WHITE = (255, 255, 255, 0)
ORANGE = (255, 100, 0, 128)
RED = (255, 0, 0)
GREEN = (0, 153, 76, 128)

edge_len_pix = 960 # This is the length of each edge in pixels
# Note that we only consider square maps, which will be drawn on square PyGame surfaces

# # Toy parameters (not yet discretized into the grid-world setting)
# positions = [12, 245, 1003, 298, 933, 393, 1100, 28, 222, 353]
# schedule = [0, 1, 0]

# Toy parameters (now discretized into the grid-world setting)
positions = [0, 1, 3, 4, 4, 1, 2, 4, 3, 1]
schedule = [0, 1, 0]

class Canvas():
    def __init__(self, num_pred = 3, num_prey = 2, map_size = 5):
        # Take resolution and number of trackers as argument

        self.num_pred = num_pred
        self.num_prey = num_prey
        self.map_size = map_size

        # Define grid locator parameter
        # Since we're dealing with a square grid-world,
        # one locator is enough for both the x- and y-coordinates

        self.locator = int(edge_len_pix/self.map_size)

        # --- Some PyGame-related initialization ---
        pygame.init()
        self.clock = pygame.time.Clock()
        self.display_surface = pygame.display.set_mode((edge_len_pix, edge_len_pix))
        pygame.display.set_caption("Predator Prey Simulator")
        self.movable_surface = pygame.Surface((edge_len_pix, edge_len_pix))
        self.message_surface = pygame.Surface((edge_len_pix, 32), pygame.SRCALPHA)
        self.message_surface = self.message_surface.convert_alpha()
        self.done_surface = pygame.Surface((edge_len_pix, edge_len_pix), pygame.SRCALPHA)
        self.done_surface = self.done_surface.convert_alpha()
        self.mx = self.movable_surface.get_width()
        self.my = self.movable_surface.get_height()
        self.done = False

        # --- Diplay screen resolution ---
        # For displaying the message from the learning module
        self.fs = 32
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), self.fs)
        
        # Frame is fixed
        self.framex = edge_len_pix
        self.framey = edge_len_pix

        # Movable surface is variable
        self.wx = self.mx
        self.wy = self.my
        self.zoom_sensitivity = 1.02 # Change this to zoom faster
        self.pan_sensitivity = 5 # Change this to move screen faster
        self.sx = 0
        self.sy = 0

        # --- Testing for scroll ---
        self.tx = 0
        self.ty = 0
        
        self.center_mark_size_px = 10
        self.center_mark_thickness_px = 1
        self.button_size_px = 50
        
        self.guiObjectsList = []

        # Some viewing margin for the button spacing
        self.vmargin = 5

        # Correctors for intuitive viewing
        self.angle_corrector = 90
        self.x_corrector = self.mx/2
        self.y_corrector = self.my/2
        self.cam_view_scaler = 2

        self.button_value = -1
        
    def setup(self):

        # --- guiObjects setup ---
        # Randomly positioned for now... get real values later
        self.target_cnt = self.num_prey # Allow only one target
        self.target_size_px = 20 # The size of the target in pixels

        self.btn_pause_surface = pygame.Surface((self.button_size_px, self.button_size_px), pygame.SRCALPHA)
        self.btn_pause_surface = self.btn_pause_surface.convert_alpha()
        self.btn_pause_surface.fill(WHITE)

        self.btn_play_surface = pygame.Surface((self.button_size_px, self.button_size_px), pygame.SRCALPHA)
        self.btn_play_surface = self.btn_play_surface.convert_alpha()
        self.btn_play_surface.fill(WHITE)

        self.btn_ff_surface = pygame.Surface((self.button_size_px, self.button_size_px), pygame.SRCALPHA)
        self.btn_ff_surface = self.btn_ff_surface.convert_alpha()
        self.btn_ff_surface.fill(WHITE)
        
        self.button_press_reactor = {"pause":0, "play":0, "ff":0}


        
        # Append the predators first and then the preys
        for i in range(self.num_pred):
            self.pred = guiPred(pred_id = i)
            self.pred.setup()
            self.guiObjectsList.append(self.pred)

        for j in range(self.num_prey):
            self.prey = guiPrey(prey_id = j)
            self.prey.setup()
            self.guiObjectsList.append(self.prey)
        # guiObjectsList looks like this [pred0, pred1, ..., pred(num_pred-1), prey0, prey1, ..., prey(num_prey-1)]

        
    def button(self, text, bx, by, bw, bh, ac, ic, surface):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if bx + bw > mouse[0] > bx and by + bh > mouse[1] > by:
            pygame.draw.rect(surface, ac, (bx, by, self.button_size_px, self.button_size_px))
            if click[0] == 1:
                pygame.draw.rect(surface, (255, 255, 0, 128), (bx, by, self.button_size_px, self.button_size_px))
        else:
            pygame.draw.rect(surface, ic, (bx, by, self.button_size_px, self.button_size_px))

        button_font = pygame.font.SysFont(pygame.font.get_default_font(), 20)
        button_label = button_font.render(text, True, (0, 0, 0))
        surface.blit(button_label, (self.button_size_px/2 - button_font.size(text)[0]/2, self.button_size_px/2 - button_font.size(text)[1]/2))
        self.display_surface.blit(surface, (bx, by))
            
    def make_border(self, obj):

        pygame.draw.rect(obj.surface, obj.border_color, [0, 0, obj.sy, obj.border_thickness])
        pygame.draw.rect(obj.surface, obj.border_color, [0, obj.sy - obj.border_thickness, obj.sy, obj.border_thickness])
        pygame.draw.rect(obj.surface, obj.border_color, [0, 0, obj.border_thickness, obj.sy])
        pygame.draw.rect(obj.surface, obj.border_color, [obj.sx - obj.border_thickness, 0, obj.border_thickness, obj.sy])
                    
    def draw(self, positions, schedule, msg=None, done=False):
        # positions is a list of x, y describing the x, y coordinates of each agent
        # schedule is a list whose elements are either 0 or 1, signifying that the agent is scheduled (1) or not (0)
        # while not self.done:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if ((event.type == pygame.QUIT) or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_q))):
                self.done = True
            if event.type == pygame.MOUSEBUTTONDOWN:

                # --- Buttons ---
                if self.framey - self.button_size_px - self.vmargin < mouse_pos[1] < self.framey - self.vmargin:
                    # Pause button
                    if self.vmargin < mouse_pos[0] < self.vmargin + self.button_size_px:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.button_press_reactor["pause"] = min(255, self.button_press_reactor["pause"] + 200)
                            # sent = self.conn.send("pause") # TODO
                            self.button_value = 0


                    # Play button
                    if 2*self.vmargin + self.button_size_px < mouse_pos[0] < 2*self.vmargin + 2*self.button_size_px:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.button_press_reactor["play"] = min(255, self.button_press_reactor["play"] + 200)
                            # sent = self.conn.send("play") # TODO
                            self.button_value = 1


                    # Fast-forward button
                    if 3*self.vmargin + 2*self.button_size_px < mouse_pos[0] < 3*self.vmargin + 3*self.button_size_px:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.button_press_reactor["ff"] = min(255, self.button_press_reactor["ff"] + 200)
                            # sent = self.conn.send("ff") # TODO
                            self.button_value = 2

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.wx *= self.zoom_sensitivity
                    self.wy *= self.zoom_sensitivity
                if event.button == 5:
                    self.wx /= self.zoom_sensitivity
                    self.wy /= self.zoom_sensitivity

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_w]: self.sy += self.pan_sensitivity
        if pressed[pygame.K_s]: self.sy -= self.pan_sensitivity
        if pressed[pygame.K_a]: self.sx += self.pan_sensitivity
        if pressed[pygame.K_d]: self.sx -= self.pan_sensitivity

        # --- Fill background ---
        self.display_surface.fill(GREY)
        self.movable_surface.fill((255, 255, 255, 0))
        self.message_surface.fill(GREY)
        self.done_surface.fill((255, 255, 0, 128))
        self.btn_pause_surface.fill((0+self.button_press_reactor["pause"], 153, 76, 128))
        self.btn_play_surface.fill((0+self.button_press_reactor["play"], 153, 76, 128))
        self.btn_ff_surface.fill((0+self.button_press_reactor["ff"], 153, 76, 128))

        for button in self.button_press_reactor:
            self.button_press_reactor[button] = max(0, self.button_press_reactor[button]-1)



        # --- Position update ----------------------------------------------
        # Call some get_pos() function here by asking the Environment
        # Then, update the guiObjects' positions accordingly

        # RECV_UPDATE() function runs on its own thread now.
        # This is to accept asynchronous inputs from
        # (i) remote server and (ii) local keyboard input for zooming/panning.

        for obj in self.guiObjectsList:
            # Fill the surface of target and drone objects
            if (("predator" in obj.name) or (obj.name == "prey")):
                obj.surface.fill(WHITE)

        # --- guiObject update ---

        # Re-draw target circle
        for obj in self.guiObjectsList:
            if obj.name == "prey":
                obj.surface = pygame.transform.scale(obj.surface, (int(2*obj.z), int(2*obj.z)))
                pygame.draw.circle(obj.surface, obj.color, (int(obj.z), int(obj.z)), int(obj.z), 0)

        # Re-draw drone objects
        cnt = 0
        for obj in self.guiObjectsList:
            if "predator" in obj.name:
                # Re-scale each surface so that each guiObject can fit in it
                obj.surface = pygame.transform.scale(obj.surface, (int(2*obj.z), int(2*obj.z)))

                # Re-draw objects according to z-coordinate (their size will vary)
                pygame.draw.circle(obj.surface, obj.body_color, (int(obj.z), int(obj.z)), int(obj.z), 0)

                if schedule[cnt] == 1:
                    pygame.draw.circle(obj.surface, obj.eye_color, (int(obj.z), int(obj.z)), int(obj.z), 3)
                cnt += 1
        # --- Canvas update ---
        # Re-drawing is called "blitting"!

        # Blit hierarchy follows this order:
        # [BOTTOM LEVEL] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< [TOP LEVEL]
        # guiObject.label <<< guiObject.surface <<< movable_surface <<< display_surface

        # guiObject.surface.blit(guiObject.label, [position]) : write label on object's surface
        # movable_surface.blit(guiObject.surface, [position]) : draw object's surface onto movable surface
        # display_surface.blit(movable_surface, [position])   : draw movable surface onto a position-fixed display surface

        # Bottom-level blit
        for guiObject in self.guiObjectsList:
            # Target
            if guiObject.name == "prey":
                guiObject.surface.blit(guiObject.label, (int(guiObject.sx/2 - guiObject.font.size(guiObject.text)[0]/2), int(guiObject.sy/2 - guiObject.font.size(guiObject.text)[1]/2)))

            # Drones
            elif "predator" in guiObject.name:
                guiObject.surface.blit(guiObject.label, (int(guiObject.z - guiObject.font.size(guiObject.text)[0]/2), int(guiObject.z - guiObject.font.size(guiObject.text)[1]/2)))

        # Note that we no longer have updates from the socket
        # Now we have positions updated directly from the vector POSITIONS
        # Positions update before blitting onto the movable surface
        for i in range(len(self.guiObjectsList)):
            self.guiObjectsList[i].x = positions[2*i]*self.locator + int(self.locator/2)
            self.guiObjectsList[i].y = positions[2*i+1]*self.locator + int(self.locator/2)

        # Mid-level blit

        # Writing the message onto the message surface
        self.text = str(msg)
        self.label = self.font.render(self.text, True, (255, 255, 255))
        self.message_surface.blit(self.label, (8, 4))

        # Draw the grid lines
        for i in range(self.map_size):
            pygame.draw.line(self.movable_surface, GREY, (i*self.locator, 0), (i*self.locator, edge_len_pix))
            pygame.draw.line(self.movable_surface, GREY, (0, i*self.locator), (edge_len_pix, i*self.locator))

        for obj in self.guiObjectsList:
            # Target and Drones
            if obj.name == "prey":
                self.movable_surface.blit(obj.surface, (int(obj.x - obj.z), int(obj.y - obj.z)))
            elif "predator" in obj.name:
                self.movable_surface.blit(obj.surface, (int(obj.x - obj.z), int(obj.y - obj.z)))
            elif (obj.name == "center"):
                self.movable_surface.blit(obj.surface, (int(self.framex/2 - obj.sx/2), int(self.framey/2 - obj.sy/2)))

        # Top-level blit

        # Blitting the movable surface onto the display surface
        self.display_surface.blit(pygame.transform.scale(self.movable_surface, (int(self.wx), int(self.wy))), (int((self.framex - self.wx)/2 + self.sx), int((self.framey - self.wy)/2 + self.sy)))

        # Blitting the message surface onto the display surface
        self.display_surface.blit(self.message_surface, (0, 0))

        if done:
            self.display_surface.blit(self.done_surface, (0, 0))
            sleep(1)

        # Re-draw buttons
        self.button("PAUSE", self.vmargin, self.framey - self.vmargin - self.button_size_px, self.button_size_px, self.button_size_px, (0, 255, 0, 128), GREEN, self.btn_pause_surface)
        self.button("PLAY", 2*self.vmargin + self.button_size_px, self.framey - self.vmargin - self.button_size_px, self.button_size_px, self.button_size_px, (0, 255, 0, 128), GREEN, self.btn_play_surface)
        self.button("FF", 3*self.vmargin + 2*self.button_size_px, self.framey - self.vmargin - self.button_size_px, self.button_size_px, self.button_size_px, (0, 255, 0, 128), GREEN, self.btn_ff_surface)

        pygame.display.update()

        if self.button_value == 1:
            sleep(0.5)
        elif self.button_value == 0:
            self.button_value = 3
            while self.button_value == 3:
                sleep(0.1)
                mouse_pos = pygame.mouse.get_pos()
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:

                        # --- Buttons ---
                        if self.framey - self.button_size_px - self.vmargin < mouse_pos[1] < self.framey - self.vmargin:
                            # Pause button
                            if self.vmargin < mouse_pos[0] < self.vmargin + self.button_size_px:
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    self.button_press_reactor["pause"] = min(255, self.button_press_reactor["pause"] + 200)
                                    # sent = self.conn.send("pause") # TODO
                                    self.button_value = 0

                            # Play button
                            if 2 * self.vmargin + self.button_size_px < mouse_pos[
                                0] < 2 * self.vmargin + 2 * self.button_size_px:
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    self.button_press_reactor["play"] = min(255, self.button_press_reactor["play"] + 200)
                                    # sent = self.conn.send("play") # TODO
                                    self.button_value = 1

                            # Fast-forward button
                            if 3 * self.vmargin + 2 * self.button_size_px < mouse_pos[
                                0] < 3 * self.vmargin + 3 * self.button_size_px:
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    self.button_press_reactor["ff"] = min(255, self.button_press_reactor["ff"] + 200)
                                    # sent = self.conn.send("ff") # TODO
                                    self.button_value = 2


        return 0





        
if __name__ == "__main__":
    canvas = Canvas(3, 2, 8)
    canvas.setup()
    canvas.draw(positions, schedule)
        
