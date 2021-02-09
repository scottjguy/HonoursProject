import os
from math import sin, radians, degrees, copysign

import pygame
from pygame.math import Vector2
import math

from Car import Car


class Game:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Honours Project")
        width = 500
        height = 1200
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def show_text(self, text, x, y):
        font = pygame.font.Font(None, 36)
        text = font.render(text, 1, (255, 255, 255))
        self.screen.blit(text, (x, y))

    def render_cars(self, x, y, x1, y2, distance, crash_counter, current_state, episode):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 0, 0), (x, y, 20, 40))
        pygame.draw.rect(self.screen, (0, 255, 0), (x1, y2, 20, 40))

        distance_text = "Distance = " + str(distance)
        self.show_text(distance_text, 0, 0)

        crash_counter_text = "Fails = " + str(crash_counter)
        self.show_text(crash_counter_text, 0, 20)

        current_state_text = "State: " + current_state
        self.show_text(current_state_text, 0, 40)

        current_episode_text = "Episode: " + str(episode+1)
        self.show_text(current_episode_text, 0, 60)

        pygame.display.flip()

        self.clock.tick(self.ticks)

    def training_screen(self, episode, distance):
        self.screen.fill((0, 0, 0))
        episode_text = "Training episodes: " + str(episode+1) + " - " + str((episode + distance) - 1)
        self.show_text(episode_text, 0, 0)

        pygame.display.flip()
        self.clock.tick(self.ticks)

    def close(self):
        pygame.quit()

    def run(self):
        lead_car = Car(150, 100)
        following_car = Car(150, 150)
        ppu = 32

        crash_counter = 0

        while not self.exit:
            dt = self.clock.get_time() / 1000

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()

            # Logic Leading Car
            if pressed[pygame.K_UP]:
                lead_car.action(1)
            elif pressed[pygame.K_DOWN]:
                lead_car.action(0)
            else:
                lead_car.acceleration = 0

            distance = (math.sqrt((following_car.position_x - lead_car.position_x) ** 2 +
                                  (following_car.position_y - lead_car.position_y) ** 2)) - 40

            # Logic Following Car
            if distance > 40:
                following_car.action(1)
            elif distance < 40:
                following_car.action(0)
            else:
                following_car.acceleration = 0

            if distance <= 0 or distance >= 60:
                crash_counter += 1
                lead_car.position_x = 150
                lead_car.position_y = 100
                following_car.position_x = 150
                following_car.position_y = 150

            # Drawing
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (255, 0, 0), (lead_car.position_x, lead_car.position_y, 20, 40))
            pygame.draw.rect(self.screen, (0, 255, 0), (following_car.position_x, following_car.position_y, 20, 40))

            distance_text = "Distance = " + str(distance)
            self.show_text(distance_text, 0, 0)

            crash_counter_text = "Fails = " + str(crash_counter)
            self.show_text(crash_counter_text, 0, 20)

            pygame.display.flip()

            self.clock.tick(self.ticks)
        pygame.quit()


