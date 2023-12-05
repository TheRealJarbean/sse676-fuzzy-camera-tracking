import keyboard
from gpiozero import Servo
from time import sleep

while(True):
    selection = input('Input servo to test. (pan/tilt): ')
    if selection == 'tilt':
        servo = Servo(17)
        break
    elif selection == 'pan':
        servo = Servo(27)
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
