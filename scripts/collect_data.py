import sys
import os
import json
from time import time
from datetime import datetime
import csv
import serial
import pygame
import cv2 as cv
from picamera2 import Picamera2
from gpiozero import LED

# SETUP
# Load configs
params_file_path = os.path.join(sys.path[0], 'configs.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
THROTTLE_AXIS = params['throttle_joy_axis']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']
# Init LED
headlight = LED(params['led_pin'])
headlight.off()

# Init serial port
ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
print(f"Pico is connected to port: {ser_pico.name}")

# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()

# Create data directory
image_dir = os.path.join(
    os.path.dirname(sys.path[0]),
    'data', datetime.now().strftime("%Y-%m-%d-%H-%M"),
    'images/'
)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')

# Init camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": 'RGB888', "size": (176, 208)},
        controls={"FrameDurationLimits": (50000, 50000)},  # 20 FPS
    )
)
cam.start()

# Init timer for FPS computing
start_stamp = time()
frame_counts = 0

# Init variables
ax_val_st = 0.0  # center steering
ax_val_th = 0.0  # shut throttle
is_recording = False

# Countdown for camera warm-up
for i in reversed(range(60)):
    frame = cam.capture_array()
    if frame is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % 20:
        print(i // 20)  # count down 3, 2, 1 sec

# LOOP
try:
    while True:
        frame = cam.capture_array()  # read image
        if frame is None:
            print("No frame received. TERMINATE!")
            headlight.close()
            cv.destroyAllWindows()
            pygame.quit()
            ser_pico.close()
            sys.exit()

        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)  # keep 2 decimals
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)  # keep 2 decimals
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    is_recording = not is_recording
                    print(f"Recording: {is_recording}")
                    headlight.toggle()
                elif js.get_button(STOP_BUTTON):  # emergency stop
                    print("E-STOP PRESSED. TERMINATE!")
                    headlight.off()
                    headlight.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    ser_pico.close()
                    sys.exit()

        # Calculate steering and throttle value
        act_st = ax_val_st  # steer action: -1: left, 1: right
        act_th = ax_val_th  # throttle action: -1: max backward, 1: max forward

        # Encode duty cycles in milliseconds
        duty_st = int((act_st + 1) * 500 + 1000)  # Example conversion
        duty_th = int((act_th + 1) * 500 + 1000)  # Example conversion

        msg = f"{duty_st},{duty_th}\n".encode('utf-8')
        ser_pico.write(msg)

        # Log data
        action = [act_st, act_th]
        if is_recording:
            cv.imwrite(image_dir + str(frame_counts) + '.jpg', frame)
            label = [str(frame_counts) + '.jpg'] + action
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)

        frame_counts += 1

        # Log frame rate
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")

        # Press "q" to quit
        if cv.waitKey(1) == ord('q'):
            headlight.off()
            headlight.close()
            cv.destroyAllWindows()
            pygame.quit()
            ser_pico.close()
            sys.exit()

# Handle terminate signal (Ctrl-c)
except KeyboardInterrupt:
    headlight.off()
    headlight.close()
    cv.destroyAllWindows()
    pygame.quit()
    ser_pico.close()
    sys.exit()
