import os
from math import sin, radians, degrees, copysign
import math

import pygame
from pygame.math import Vector2


class Car:
    def __init__(self, x, y, angle=0.0, length=4, max_steering=30, max_acceleration=1.0):

        # default variables for all cars
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 2
        self.free_deceleration = 1000

        # individual variable for specific instance
        self.acceleration = 0.0
        self.steering = 0.0  # steering being applied
        self.position = Vector2(x, y)  # position of the centre of the vehicle
        self.angle = angle  # direction the vehicle is facing
        self.velocity = Vector2(0.0, 0.0)

        self.position_fmiddle = Vector2(x, y)
        self.position_end_sensor = Vector2(x, y)
        self.sensor_angle = -50.0

        # Sensor Co-ordinates
        self.position_middle = Vector2(x, y)
        self.position_left = Vector2(x, y)
        self.position_right = Vector2(x, y)

        self.position_back_left = Vector2(x, y)
        self.position_back_right = Vector2(x, y)
        self.r_angle_middle = 0.0
        self.r_angle_left = 0.0
        self.r_angle_right = 0.0

        self.r_angle_back_right = 0.0
        self.r_angle_back_left = 0.0
        self.r_angle_fright = 0.0
        self.r_sensor_angle = 0.0
        self.r_angle_fmiddle = 0.0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        if self.acceleration == 0:
            self.velocity.x = 0
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        # updating vehicles position and rotation
        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt

        if self.angle >= 180:
            self.angle = -180
        elif self.angle <= -180:
            self.angle = 180

        # Sensor logic when steering

        # Front of vehicle Sensor

        self.r_angle_fmiddle = self.angle * (math.pi / 180)

        self.position_fmiddle.x = self.position.x + (2 * math.cos(-self.r_angle_fmiddle))
        self.position_fmiddle.y = self.position.y + (2 * math.sin(-self.r_angle_fmiddle))

        # End of Scanner Sensor

        self.r_sensor_angle = (self.angle - self.sensor_angle) * (math.pi / 180)

        self.position_end_sensor.x = self.position.x + (6 * math.cos(-self.r_sensor_angle))
        self.position_end_sensor.y = self.position.y + (6 * math.sin(-self.r_sensor_angle))

        # Back-left corner (Bumper)

        self.r_angle_back_left = (self.angle - 210) * (math.pi / 180)

        self.position_back_left.x = self.position.x + (2 * math.cos(-self.r_angle_back_left))
        self.position_back_left.y = self.position.y + (2 * math.sin(-self.r_angle_back_left))

        # Back-middle Sensor (Bumper)
        self.r_angle_middle = (self.angle - 180) * (math.pi / 180)

        self.position_middle.x = self.position.x + (2 * math.cos(-self.r_angle_middle))
        self.position_middle.y = self.position.y + (2 * math.sin(-self.r_angle_middle))

        # Back-right corner (Bumper)

        self.r_angle_back_right = (self.angle - 150) * (math.pi / 180)

        self.position_back_right.x = self.position.x + (2 * math.cos(-self.r_angle_back_right))
        self.position_back_right.y = self.position.y + (2 * math.sin(-self.r_angle_back_right))

    def update_sensor(self):
        self.r_sensor_angle = (self.angle - (self.sensor_angle)) * (math.pi / 180)

        self.position_end_sensor.x = self.position.x + (6 * math.cos(-self.r_sensor_angle))
        self.position_end_sensor.y = self.position.y + (6 * math.sin(-self.r_sensor_angle))

    def action(self, act, dt):
        # Forward movement
        if act == 0:
            # Check if the vehicle is going backwards
            if self.velocity.x < 0:
                self.acceleration = self.free_deceleration
            else:
                # increase the acceleration
                self.acceleration += 1 * dt

        # Backward Movement
        elif act == 1:
            # Check if the vehicle is going forwards
            if self.velocity.x > 0:
                self.acceleration = -self.free_deceleration
            else:
                # Decrease the acceleration
                self.acceleration -= 1 * dt

        # Do nothing in terms of speed
        elif act == 2:
            if dt != 0:
                # set acceleration to 0 (No Movement)
                self.acceleration = 0
            if self.acceleration == 0:
                self.velocity.x = 0

        # Steering left
        elif act == 3:
            if self.steering >= 0:
                self.steering += 30 * dt
            elif self.steering < 0:
                self.steering = 0

        # Steering Right
        elif act == 4:
            if self.steering <= 0:
                self.steering -= 30 * dt
            elif self.steering > 0:
                self.steering = 0

        # No Steering
        elif act == 5:
            self.steering = 0
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
