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

    def cal_angle(self, x1, y1, x2, y2):
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

    def render_information(self, distance, crash_counter, steering_type, episode, velocity, velocity1):
        distance_text = "Distance = " + str(distance)
        self.show_text(distance_text, 0, 0)

        crash_counter_text = "Fails = " + str(crash_counter)
        self.show_text(crash_counter_text, 0, 30)

        current_state_text = "Steering Type: " + steering_type
        self.show_text(current_state_text, 0, 60)

        current_steering_text = "Steering: " + episode
        self.show_text(current_steering_text, 0, 90)

        current_velocity_text = "Lead Velocity: " + str(velocity)
        self.show_text(current_velocity_text, 0, 120)

        # ---------------------------------------------------------------

        current_velocity_text = "Following Velocity: " + str(velocity1)
        self.show_text(current_velocity_text, 450, 120)

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

        # AI state
        learning_state = True

        # file name goes here for existing q table
        speed_q_table = "qtableSpeed-1586792839.pickle"

        steering_q_table = "qtableSensorSteering-1587931416.pickle"

        if speed_q_table is None:
            print("Train the speed first")
        else:
            with open(speed_q_table, "rb") as f:
                q_table_speed = pickle.load(f)

        if steering_q_table is None:
            print("Train the steering first")
        else:
            with open(steering_q_table, "rb") as f:
                q_table_steering = pickle.load(f)

        while not self.exit:

            if learning_state:
                state = "Wide Sensor Steering"
                steering_dir = "No Movement"


                for episode in range(10):

                    lead_car = Car(10, 12)
                    follow_car = Car(5, 12)
                    show = True

                    for i in range(15000):
                        dt = self.clock.get_time() / 1000

                        # Event queue
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.exit = True

                        speed_obs = int((self.cal_distance(lead_car.position_middle.x, lead_car.position_middle.y,
                                                           follow_car.position_fmiddle.x,
                                                           follow_car.position_fmiddle.y) * 1000))

                        speed_action = np.argmax(q_table_speed[speed_obs])

                        if speed_obs <= 50:
                            crash_counter += 1
                            break

                        pressed = pygame.key.get_pressed()

                        if pressed[pygame.K_UP]:
                            lead_car.action(0, dt)
                            follow_car.action(speed_action, dt)
                        elif pressed[pygame.K_DOWN]:
                            lead_car.action(1, dt)
                            follow_car.action(speed_action, dt)
                        elif pressed[pygame.K_SPACE]:
                            lead_car.action(4, dt)
                        else:
                            lead_car.action(2, dt)
                            follow_car.action(speed_action, dt)

                        lead_car.acceleration = max(-lead_car.max_acceleration,
                                                    min(lead_car.acceleration, lead_car.max_acceleration))
                        follow_car.acceleration = max(-follow_car.max_acceleration,
                                                      min(follow_car.acceleration, follow_car.max_acceleration))
                        if pressed[pygame.K_RIGHT]:
                            lead_car.action(4, dt)
                        elif pressed[pygame.K_LEFT]:
                            lead_car.action(3, dt)
                        else:
                            lead_car.action(5, dt)
                        lead_car.steering = max(-lead_car.max_steering, min(lead_car.steering, lead_car.max_steering))

                        follow_car.update(dt)
                        lead_car.update(dt)

                        # -------------------Steering-------------------
                        if follow_car.velocity.x != 0:

                            angle = self.cal_angle(follow_car.position_fmiddle.x, follow_car.position_fmiddle.y,
                                                   lead_car.position.x, lead_car.position.y)

                            pygame.draw.rect(self.screen, (255, 0, 0),
                                             (lead_car.position_fmiddle.x * ppu, lead_car.position_fmiddle.y * ppu, 5,
                                              5))

                            # print(angle)
                            # print((follow_car.angle + 90))
                            steering_obs = int(self.ult_steering(follow_car, lead_car) + 180)

                            # print("leading car")
                            # print(lead_car.angle)

                            action = np.argmax(q_table_steering[steering_obs])

                            # If vehicle is traveling backwards then reverse the steering
                            if follow_car.velocity.x < 0:
                                if action == 0:
                                    action = 1
                                elif action == 1:
                                    action = 0
                                else:
                                    action = 2

                            follow_car.action(action + 3, dt)
                            follow_car.steering = max(-follow_car.max_steering,
                                                      min(follow_car.steering, follow_car.max_steering))
                            # print(steering_obs)

                            steering_dir = ""

                            if int(self.ult_steering(follow_car, lead_car)) < 0:
                                steering_dir = "left"
                            elif int(self.ult_steering(follow_car, lead_car)) > 0:
                                steering_dir = "right"
                            else:
                                steering_dir = "straight ahead"

                            lead_car.update(dt)
                            follow_car.update(dt)

                        # --------------------------------------User input--------------------------------------

                        # Controls the steering of the vehicle

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

                            self.render_information(speed_obs, crash_counter, state, steering_dir, lead_car.velocity.x,
                                                    follow_car.velocity.x)

                            pygame.display.update()

                            self.clock.tick(self.ticks)
                self.exit = True

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
