import os
from math import sin, radians, degrees, copysign

import numpy as np
import pygame
from pygame.math import Vector2
import math
import pickle
import time

from Car import Car


class Game:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Honours Project")
        width = 1500
        height = 1000
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    @staticmethod
    def cal_angle(x1, y1, x2, y2):
        delta_x = x2 - x1
        delta_y = y2 - y1

        degrees_temp = np.math.atan2(delta_x, delta_y) / math.pi * 180

        return degrees_temp

    def show_text(self, text, x, y):
        font = pygame.font.Font(None, 36)
        text = font.render(text, 1, (255, 255, 255))
        self.screen.blit(text, (x, y))

    def cal_distance(self, x1, y1, x2, y2):
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

    def ult_steering(self, c1, c2,):

        found = False
        counter_left = 0
        counter_right = 0
        ppu = 32
        sensor = 50

        for i in range((sensor * 2) + 1):

            p0_x = c1.position_fmiddle.x
            p0_y = c1.position_fmiddle.y

            p1_x = c1.position_end_sensor.x
            p1_y = c1.position_end_sensor.y

            p2_x = c2.position_back_left.x
            p2_y = c2.position_back_left.y

            p3_x = c2.position_back_right.x
            p3_y = c2.position_back_right.y

            A1 = p1_y - p0_y
            B1 = p0_x - p1_x
            C1 = A1 * p0_x + B1 * p0_y
            A2 = p3_y - p2_y
            B2 = p2_x - p3_x
            C2 = A2 * p2_x + B2 * p2_y
            denominator = A1 * B2 - A2 * B1

            if denominator == 0:
                return 9999999999999999

            intersect_x = (B2 * C1 - B1 * C2) / denominator
            intersect_y = (A1 * C2 - A2 * C1) / denominator

            if p1_x - p0_x == 0:
                rx0 = 2
            else:
                rx0 = (intersect_x - p0_x) / (p1_x - p0_x)

            if p1_y - p0_y == 0:
                ry0 = 2
            else:
                ry0 = (intersect_y - p0_y) / (p1_y - p0_y)

            if p3_x - p2_x == 0:
                rx1 = 2
            else:
                rx1 = (intersect_x - p2_x) / (p3_x - p2_x)

            if p3_y - p2_y == 0:
                ry1 = 2
            else:
                ry1 = (intersect_y - p2_y) / (p3_y - p2_y)

            if((0 <= rx0 <= 1) or (0 <= ry0 <= 1)) and ((0 <= rx1 <= 1) or (0 <= ry1 <= 1)):
                found = True

                intersect_x = (B2 * C1 - B1 * C2) / denominator
                intersect_y = (A1 * C2 - A2 * C1) / denominator

                pygame.draw.rect(self.screen, (255, 0, 0), (intersect_x * ppu, intersect_y * ppu, 5, 5))
                pygame.draw.line(self.screen, (255, 0, 0),
                                 (p2_x * ppu, p2_y * ppu),
                                 (p3_x * ppu, p3_y * ppu), 1)
                pygame.draw.line(self.screen, (255, 0, 0),
                                 (p0_x * ppu, p0_y * ppu),
                                 (p1_x * ppu, p1_y * ppu),
                                 1)

            else:
                # print("Not connect")

                if not found:
                    pygame.draw.line(self.screen, (0, 255, 0),
                                 (p0_x * ppu, p0_y * ppu),
                                 (p1_x * ppu, p1_y * ppu),
                                 1)
                    counter_left += 1
                else:
                    pygame.draw.line(self.screen, (0, 0, 255),
                                     (p0_x * ppu, p0_y * ppu),
                                     (p1_x * ppu, p1_y * ppu),
                                     1)
                    counter_right += 1
            pygame.display.update()

            c1.sensor_angle += 1
            c1.update_sensor()

        c1.sensor_angle = -sensor
        c1.update_sensor()

        if not found:
            return 0
        else:
            return counter_left - counter_right

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        ppu = 32

        crash_counter = 0

        ANGLE_MAX = 8000
        ANGLE_IDEAL = 0

        # rewards
        HM_EPISODES = 1000
        CRASH_PENALTY = -300
        ANGLE_REWARD = 5000

        # q learning Variables
        epsilon = 0
        LEARNING_RATE = 0.1
        DISCOUNT = 0.95

        # AI state
        learning_state = True

        set_reward = False

        steering_q_table = "qtableSteering-1586265152.pickle"

        if steering_q_table is None:
            q_table_steering = np.zeros((30000, 3))
        else:
            with open(steering_q_table, "rb") as f:
                q_table_steering = pickle.load(f)

        while not self.exit:

            if learning_state:
                state = "Exploring"

                for episode in range(HM_EPISODES):
                    if not set_reward:
                        random_y = 0
                        set_reward = True
                    else:
                        random_y = np.random.randint(-30, 30)

                    lead_car = Car(12, 12, angle=random_y)
                    follow_car = Car(5, 12)

                    print(f"on # {episode}, epsilon: {epsilon}")
                    show = True

                    episode_rewards = 0

                    for i in range(250):
                        dt = self.clock.get_time() / 1000

                        # Event queue
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.exit = True

                        # Vehicles going forward Automatically 
                        follow_car.action(0, dt)
                        lead_car.action(0, dt)

                        follow_car.update(dt)
                        lead_car.update(dt)

                        # -------------------Steering-------------------
                        if follow_car.velocity.x > 0:

                            angle = self.cal_angle(follow_car.position_fmiddle.x, follow_car.position_fmiddle.y,
                                                   lead_car.position.x, lead_car.position.y)

                            steering_obs = int(round(angle - (follow_car.angle + 90), 2) * 100)

                            if - 18000 > steering_obs:
                                steering_obs += 36000
                            if 18000 < steering_obs:
                                steering_obs -= 36000

                            if np.random.random() > epsilon:
                                action = np.argmax(q_table_steering[steering_obs + 18000])
                            else:
                                action = np.random.randint(0, 3)

                            # If vehicle is traveling backwards then reverse the steering
                            if follow_car.velocity.x < 0:
                                if action == 0:
                                    action = 1
                                elif action == 2:
                                    action = 1
                                else:
                                    action = 3

                            follow_car.action(action + 3, dt)
                            follow_car.update(dt)

                            angle = self.cal_angle(follow_car.position_fmiddle.x, follow_car.position_fmiddle.y,
                                                   lead_car.position.x, lead_car.position.y)

                            steering_new_obs = int(round(angle - (follow_car.angle + 90), 2) * 100)

                            if - 18000 > steering_new_obs:
                                steering_new_obs += 36000
                            if 18000 < steering_new_obs:
                                steering_new_obs -= 36000

                            if steering_new_obs <= -ANGLE_MAX or steering_new_obs >= ANGLE_MAX:
                                reward = -CRASH_PENALTY
                            elif steering_new_obs == ANGLE_IDEAL:
                                reward = ANGLE_REWARD
                            else:
                                reward = -1

                            max_future_q = np.max(q_table_steering[steering_new_obs + 18000])
                            current_q = q_table_steering[steering_obs + 18000][action - 3]

                            if reward == ANGLE_REWARD:
                                new_q = ANGLE_REWARD
                            elif reward == -CRASH_PENALTY:
                                new_q = -CRASH_PENALTY
                            else:
                                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                                            reward + DISCOUNT * max_future_q)

                            if (steering_new_obs != steering_obs and action != 5) or action == 5:
                                q_table_steering[steering_obs + 18000][action - 3] = new_q

                            if reward == -CRASH_PENALTY:
                                break

                        # --------------------------------------User input---------------------------------------

                        pressed = pygame.key.get_pressed()

                        # Controls the Acceleration, braking and reverse
                        if pressed[pygame.K_UP]:
                            lead_car.action(0, dt)
                        elif pressed[pygame.K_DOWN]:
                            lead_car.action(1, dt)
                        elif pressed[pygame.K_SPACE]:
                            lead_car.action(4, dt)
                        else:
                            lead_car.action(3, dt)

                        lead_car.acceleration = max(-lead_car.max_acceleration,
                                                    min(lead_car.acceleration, lead_car.max_acceleration))

                        # Controls the steering of th vehicle
                        if pressed[pygame.K_RIGHT]:
                            lead_car.action(4, dt)
                        elif pressed[pygame.K_LEFT]:
                            lead_car.action(3, dt)
                        else:
                            lead_car.action(5, dt)
                        lead_car.steering = max(-lead_car.max_steering, min(lead_car.steering, lead_car.max_steering))

                        lead_car.update(dt)

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

                            self.render_information(0, crash_counter, state, 0,
                                                    lead_car.velocity.x)

                            pygame.display.update()

                            self.clock.tick(self.ticks)



                with open(f"qtableSteering-{int(time.time())}.pickle", "wb") as f:
                    pickle.dump(q_table_steering, f)
                    learning_state = False
                self.exit = True

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
