#!/usr/bin/env python3
import rospy
import math
import numpy as np

from std_msgs.msg import Bool
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, VehicleSpeedRpt
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import time

class PID(object):
    def __init__(self, kp, ki, kd, wg=None):
        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):
        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # integral wind-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class LaneFollowController(object):
    def __init__(self):
        rospy.init_node('lane_follow_controller', anonymous=True)
        self.rate = rospy.Rate(10)

        # Parameters
        self.desired_speed = 3.0  # m/s, reference speed
        self.max_accel     = 4.0  # limit acceleration command
        # Image geometry parameters (adjust based on your camera resolution):
        self.image_width = 1280
        self.image_center_x = self.image_width / 2.0

        # PID for speed and steering
        # Tuning required:
        self.pid_speed = PID(kp=0.5, ki=0.0, kd=0.1, wg=20)
        self.pid_steer = PID(kp=0.01, ki=0.0, kd=0.005) # example gains for steering

        self.speed = 0.0
        self.endgoal_x = None
        self.endgoal_y = None

        # GEM vehicle enable
        self.gem_enable    = False
        self.pacmod_enable = False

        # Subscribers
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)
        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.endgoal_sub = rospy.Subscriber("/lane_detection/endgoal", PoseStamped, self.endgoal_callback)

        # Publishers
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.gear_pub   = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.brake_pub  = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.accel_pub  = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.turn_pub   = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.steer_pub  = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

        # Commands
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 3

        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0
        self.steer_cmd.angular_velocity_limit = 3.5

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3)

    def endgoal_callback(self, msg):
        self.endgoal_x = msg.pose.position.x
        self.endgoal_y = msg.pose.position.y

    def front2steer(self, f_angle):
        # same front2steer logic as original code
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle_p = -f_angle
            steer_angle = -round(-0.1084*f_angle_p**2 + 21.775*f_angle_p, 2)
        else:
            steer_angle = 0.0
        return steer_angle

    def start_control(self):
        while not rospy.is_shutdown():
            if not self.gem_enable:
                if self.pacmod_enable:
                    # Enable forward gear
                    self.gear_cmd.ui16_cmd = 3
                    # Enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0
                    # Enable accel
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    self.accel_cmd.f64_cmd = 1.5

                    print("Current Gear: ", self.gear_cmd.ui16_cmd)
                    print("Current Brake: ", self.brake_cmd.f64_cmd)
                    print("Current Accel: ", self.accel_cmd.f64_cmd)
                    print("Current Turn: ", self.turn_cmd.ui16_cmd)

                    self.gear_pub.publish(self.gear_cmd)
                    self.turn_pub.publish(self.turn_cmd)
                    self.brake_pub.publish(self.brake_cmd)
                    self.accel_pub.publish(self.accel_cmd)
                    self.gem_enable = True
                    rospy.loginfo("GEM Enabled with Forward Gear!")

            # If we have an endgoal waypoint, compute steering error
            # Lateral error: difference between endgoal_x and image center
            # Consider that each pixel might correspond to some angle. For simplicity,
            # treat error in pixels directly. You may need a scale factor depending on your setup.
            if self.endgoal_x is not None:
                lateral_error_pixels = (self.endgoal_x - self.image_center_x)
                # Convert pixel error to a front wheel angle in degrees
                # Simple approach: assume linear mapping from pixel error to steering angle.
                # Tune scaling_factor as needed.
                scaling_factor = 0.6 # degrees per pixel (example)
                desired_front_angle = -lateral_error_pixels * scaling_factor

                # Optional: use PID on pixel error if needed
                current_time = rospy.get_time()
                steering_output = self.pid_steer.get_control(current_time, lateral_error_pixels, fwd=0.0)
                # Map steering_output (some value) to a front angle
                # You could just use steering_output as degrees if tuned accordingly.
                # We'll assume steering_output itself represents degrees of front angle:
                front_angle = -steering_output * 4.0 # Another scaling for PID output
                # Limit and convert to steering wheel angle
                steering_angle = self.front2steer(front_angle)

                # Speed control
                speed_time = rospy.get_time()
                speed_output_accel = self.pid_speed.get_control(speed_time, self.desired_speed - self.speed)

                if speed_output_accel > self.max_accel:
                    speed_output_accel = self.max_accel
                if speed_output_accel < 0.2:
                    speed_output_accel = 0.2

                # Set turn signal logic if needed:
                if front_angle <= 30 and front_angle >= -30:
                    self.turn_cmd.ui16_cmd = 1
                elif front_angle > 30:
                    self.turn_cmd.ui16_cmd = 2 # turn left
                else:
                    self.turn_cmd.ui16_cmd = 0 # turn right

                self.accel_cmd.f64_cmd = speed_output_accel
                self.steer_cmd.angular_position = math.radians(steering_angle)

                if self.gem_enable:
                    rospy.loginfo(f"Lateral error: {lateral_error_pixels} px, Steering angle: {steering_angle} deg, Speed: {self.speed} m/s")
                    print((f"Lateral error: {lateral_error_pixels} px, Steering angle: {steering_angle} deg, Speed: {self.speed} m/s"))

                print("Current Gear: ", self.gear_cmd.ui16_cmd)
                print("Current Brake: ", self.brake_cmd.f64_cmd)
                print("Current Accel: ", self.accel_cmd.f64_cmd)

                self.accel_pub.publish(self.accel_cmd)
                self.steer_pub.publish(self.steer_cmd)
                self.turn_pub.publish(self.turn_cmd)

            self.rate.sleep()

if __name__ == '__main__':
    controller = LaneFollowController()
    try:
        controller.start_control()
    except rospy.ROSInterruptException:
        pass
