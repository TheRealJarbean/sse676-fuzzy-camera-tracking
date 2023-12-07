import keyboard
from gpiozero import Device, Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

min_pw = 0.9 / 1000
max_pw = 2.5 / 1000
Device.pin_factory = PiGPIOFactory()

while(True):
    selection = input('Input servo to test. (pan/tilt): ')
    if selection == 'tilt':
        servo = Servo(27, min_pulse_width=min_pw, max_pulse_width=max_pw, initial_value=1)
        break
    elif selection == 'pan':
        servo = Servo(17, min_pulse_width=min_pw, max_pulse_width=max_pw)
        break
    print('Invalid selection.')

# Servo values range from -1 (min) to 1 (max)

while(True):
    value = input('Input value to move servo (\'exit\' to stop): ')
    if value == 'exit':
        print('Exiting...')
        break
    value = float(value)
    if value >= -1 and value <= 1:
        servo.value = value
        sleep(1)
    else:
        print('Invalid value.')
    print(servo.pulse_width)
