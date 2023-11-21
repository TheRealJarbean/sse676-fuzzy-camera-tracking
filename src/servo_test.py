from gpiozero import Servo
from time import sleep

servo = Servo(17, min_pulse_width=1/1000, max_pulse_width=2/1000)

while True:
    print("max")
    servo.max()
    sleep(3)
    print("min")
    servo.min()
    sleep(3)
