import os
from math import sin, radians, degrees, copysign

import numpy as np
import pygame
from pygame.math import Vector2
import math
import pickle
import time

from Car import Car


class Q_learning_Speed:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Honours Project")
        width = 1000
        height = 1000
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def show_text(self, text, x, y):
        font = pygame.font.Font(None, 36)
        text = font.render(text, 1, (255, 255, 255))
        self.screen.blit(text, (x, y))

    # calculate the distance between the 2 vehicles

    @staticmethod
    def cal_distance(x1, y1, x2, y2):
        distance = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 5)
        return distance

    def render_information(self, distance, crash_counter, current_state, episode, velocity):
        distance_text = "Distance = " + str(distance)
        self.show_text(distance_text, 0, 0)

        crash_counter_text = "Fails = " + str(crash_counter)
        self.show_text(crash_counter_text, 0, 20)

        current_state_text = "State: " + current_state
        self.show_text(current_state_text, 0, 40)

        current_episode_text = "Episode: " + str(episode + 1)
        self.show_text(current_episode_text, 0, 60)

        current_velocity_text = "Velocity: " + str(velocity)
        self.show_text(current_velocity_text, 0, 80)

        pygame.display.flip()

        self.clock.tick(self.ticks)

    def training_screen(self, episode, distance):
        self.screen.fill((0, 0, 0))
        episode_text = "Training episodes: " + str(episode + 1) + " - " + str((episode + distance) - 1)
        self.show_text(episode_text, 0, 0)

        pygame.display.flip()
        self.clock.tick(self.ticks)

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        ppu = 32

        crash_counter = 0

        DISTANCE_MIN = 500
        DISTANCE_MAX = 5001
        DISTANCE_IDEAL_MIN = 950
        DISTANCE_IDEAL_MAX = 1050

        # rewards
        HM_EPISODES = 500
        CRASH_PENALTY = 300
        DISTANCE_REWARD = 5000

        # q learning Variables
        epsilon = 0.9
        LEARNING_RATE = 0.1
        DISCOUNT = 0.95

        # AI state
        learning_state = True

        # file name goes here for existing q table
        start_q_table = None

        if start_q_table is None:
            q_table = np.zeros((10000, 3))
            print("Created table")
        else:
            with open(start_q_table, "rb") as f:
                q_table = pickle.load(f)

        while not self.exit:

            if learning_state:
                state = "Exploring"

                for episode in range(HM_EPISODES):
                    lead_car = Car(12, 5)
                    follow_car = Car(5, 5)

                    print(f"on # {episode}, epsilon: {epsilon}")
                    show = True

                    episode_rewards = 0

                    for i in range(10000):
                        dt = self.clock.get_time() / 1000

                        # Event queue
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.exit = True

                        obs = int((self.cal_distance(lead_car.position_middle.x, lead_car.position_middle.y,
                                                     follow_car.position_fmiddle.x,
                                                     follow_car.position_fmiddle.y) * 1000))

                        if np.random.random() > epsilon:
                            action = np.argmax(q_table[obs])
                        else:
                            action = np.random.randint(0, 3)

                        follow_car.action(action, dt)

                        follow_car.update(dt)

                        new_obs = int((self.cal_distance(lead_car.position_middle.x, lead_car.position_middle.y,
                                                         follow_car.position_fmiddle.x,
                                                         follow_car.position_fmiddle.y) * 1000))

                        if new_obs <= DISTANCE_MIN or new_obs >= DISTANCE_MAX:
                            reward = -CRASH_PENALTY
                            print(new_obs)

                        elif DISTANCE_IDEAL_MIN <= new_obs <= DISTANCE_IDEAL_MAX:
                            reward = DISTANCE_REWARD
                        else:
                            reward = -1

                        max_future_q = np.max(q_table[new_obs])
                        current_q = q_table[obs][action]

                        if reward == DISTANCE_REWARD:
                            new_q = DISTANCE_REWARD
                        elif reward == -CRASH_PENALTY:
                            new_q = -CRASH_PENALTY
                        else:
                            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                        q_table[obs][action] = new_q

                        if show:
                            self.screen.fill((0, 0, 0))
                            rotated_lead = pygame.transform.rotate(car_image, lead_car.angle)
                            rotated_following = pygame.transform.rotate(car_image, follow_car.angle)

                            rect_lead = rotated_lead.get_rect()
                            rect_follow = rotated_following.get_rect()

                            self.screen.blit(rotated_lead,
                                             lead_car.position * ppu - (rect_lead.width / 2, rect_lead.height / 2))
                            self.screen.blit(rotated_following,
                                             follow_car.position * ppu - (rect_follow.width / 2, rect_follow.height /
                                                                          2))

                            self.render_information(obs, crash_counter, state, 0,
                                                    follow_car.velocity.x)
                            pygame.draw.rect(self.screen, (255, 0, 0), (lead_car.position_end_sensor.x * ppu, lead_car.position_end_sensor.y * ppu, 5, 5))

                            pygame.display.update()

                            self.clock.tick(self.ticks)

                        episode_rewards += reward

                        if reward == -CRASH_PENALTY:
                            break

                with open(f"qtableSpeed-{int(time.time())}.pickle", "wb") as f:
                    pickle.dump(q_table, f)
                    learning_state = False
                self.exit = True

        pygame.quit()


if __name__ == '__main__':
    speed_learning = Q_learning_Speed()
    speed_learning.run()