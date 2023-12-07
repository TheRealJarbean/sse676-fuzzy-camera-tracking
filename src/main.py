# Jaron Anderson
# SSE 676
# Fuzzy algorithm to adjust speed of two motors on a two-axis camera mount, 
# one for pan and one for tilt, to track a subject and keep them roughly in the center of the frame.

# IMPORTANT: cv2 defaults to BGR color space
import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep
from libcamera import Transform
from gpiozero import Device, Servo
from gpiozero.pins.pigpio import PiGPIOFactory

Device.pin_factory = PiGPIOFactory()
min_pw_pan = 1.3 / 1000
max_pw_pan = 2.05 / 1000
min_pw_tilt = 0.7 / 1000
max_pw_tilt = 2.5 / 1000
pan_servo = Servo(17, min_pulse_width=min_pw_pan, max_pulse_width=max_pw_pan, initial_value=1)
tilt_servo = Servo(27, min_pulse_width=min_pw_tilt, max_pulse_width=max_pw_tilt)

############### CONFIGURATION #####################

PREVIEW_ENABLED = True
PREVIEW_REFRESH_RATE_MS = 5
VERTICAL_ADJUST = 50 # How many pixels to shift frame based on detection area (face)
MIN_FACE_SIZE = 100 # Length of one side of a square area in pixels
SCALE_FACTOR = 1.4 # Higher values = faster, less accurate detection; keep between 1 and 1.4

# servo.value can range from -1 to 1
PAN_SERVO_MIN = -1 # Negative direction is counterclockwise
PAN_SERVO_MAX = 1
TILT_SERVO_MIN = 0.3 # Due to the motor's physical orientation, min is up, and max is down
TILT_SERVO_MAX = 1

PAN_STRENGTH = 0.06
TILT_STRENGTH = 0.01
FINE_ADJUSTMENT_MULT = 0.5 # Strength is multiplied by this value when subject is close to center of frame

############### CONFIGURATION #####################

############### SERVO FUNCTIONS ###################

def pan(value: float):
    if value == 0:
        return
    elif pan_servo.value + value < PAN_SERVO_MIN:
        pan_servo.value = PAN_SERVO_MIN
    elif pan_servo.value + value > PAN_SERVO_MAX:
        pan_servo.value = PAN_SERVO_MAX
    else:
        pan_servo.value += value

def tilt(value: float):
    if value == 0:
        return
    elif tilt_servo.value + value < TILT_SERVO_MIN:
        tilt_servo.value = TILT_SERVO_MIN
    elif tilt_servo.value + value > TILT_SERVO_MAX:
        tilt_servo.value = TILT_SERVO_MAX
    else:
        tilt_servo.value += value

def home_servos():
    # Home pan and tilt servos
    pan_servo.value = (PAN_SERVO_MAX + PAN_SERVO_MIN) / 2
    tilt_servo.value = 0.5

############### SERVO FUNCTIONS ###################

################## CENTROID CLASS #######################

class Centroid:
        def __init__(self, x = 0, y = 0):
                self.x = x
                self.y = y

################## END CENTROID CLASS ####################

################## FUZZIFICATION FUNCTIONS ####################

def fuzzify_horizontal(centroid):
        def exiting_left(x):
                if x <= 720:
                        return 1
                elif 720 < x <= 960:
                        return (960 - x) / 240
                return 0

        def not_exiting(x):
                if 720 <= x <= 960:
                        return (x - 720) / 240
                elif 960 < x <= 1200:
                        return (1200 - x) / 240
                return 0
        
        def exiting_right(x):
                if x >= 1200:
                        return 1
                elif 960 < x < 1200:
                        return (x - 960) / 240
                return 0
        
        membership_functions = [exiting_left, not_exiting, exiting_right]
        return [func(centroid.x) for func in membership_functions]

def fuzzify_vertical(centroid):
        def exiting_top(y):
                if y <= 270:
                        return 1
                elif 270 < y <= 405:
                        return (405 - y) / 135
                return 0

        def not_exiting(y):
                if 405 <= y <= 675:
                        return 1
                elif 270 <= y < 405:
                        return (y - 270) / 135
                elif 675 < y <= 810:
                        return (810 - y) / 135
                return 0
        
        def exiting_bottom(y):
                if y >= 810:
                        return 1
                elif 675 <= y < 810:
                        return (y - 675) / 135
                return 0
        
        membership_functions = [exiting_top, not_exiting, exiting_bottom]
        return [func(centroid.y) for func in membership_functions]
                

################## END FUZZIFICATION FUNCTIONS ######################

################## DEFUZZIFICATION FUNCTIONS ########################

def defuzzify_horizontal(membership_values):
        # membership_values[0] = exiting_left
        # membership_values[1] = not_exiting
        # membership_values[2] = exiting_right

        if membership_values[1] > 0.5:
                if membership_values[1] > 0.8:
                        # Do nothing
                        print('Holding')
                        return
                if membership_values[0] > membership_values[2]:
                    print('Soft right')
                    pan(-PAN_STRENGTH * FINE_ADJUSTMENT_MULT)
                    return
                print('Soft left')
                pan(PAN_STRENGTH * FINE_ADJUSTMENT_MULT)
                return       
        elif membership_values[0] > 0.5:
                print('Hard right')
                pan(-PAN_STRENGTH)
                return
        else:
                print('Hard left')
                pan(PAN_STRENGTH)
                return

def defuzzify_vertical(membership_values):
        # membership_values[0] = exiting_top
        # membership_values[1] = not_exiting
        # membership_values[2] = exiting_bottom

        if membership_values[1] > 0.5:
                if membership_values[1] > 0.8:
                        # Do nothing
                        return
                if membership_values[0] > membership_values[2]:
                    pan(-TILT_STRENGTH * FINE_ADJUSTMENT_MULT)
                    return
                pan(TILT_STRENGTH * FINE_ADJUSTMENT_MULT)
                return
        elif membership_values[0] > 0.5:
                tilt(-TILT_STRENGTH)
                return
        else:
                tilt(TILT_STRENGTH)
                return

################## END DEFUZZIFICATION FUNCTIONS ####################

# Load trained XML classifier for detecting upper body
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create object of the Centroid class to track the upper body of subject
# Initialize to the center of the frame
upperbody_centroid = Centroid(1920 / 2, 1080 / 2)

# Establish connection to video
camera = Picamera2()
config = camera.create_video_configuration(main={"size":(1920, 1080)})
camera.configure(config)
camera.start()
sleep(1)

home_servos()
sleep(5)

frame_count = 0
dropped_frames = 0

# Loop forever
while(True):
        # Capture a frame and process it
        frame = camera.capture_array('main')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert color space for simpler OpenCV processing
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect upper bodies in frame
        upperbodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), maxSize=(300, 300))

        if len(upperbodies) == 0:
            upperbody_centroid.x = 1920 / 2
            upperbody_centroid.y = 1080 / 2
        else:
            for (x,y,w,h) in upperbodies:
                # Draw rectangle on detected upperbodies
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                upperbody_centroid.x = x + (w / 2)
                upperbody_centroid.y = y + (h / 2) + VERTICAL_ADJUST

        # Draw circle on centroid
        cv2.circle(frame, (int(upperbody_centroid.x), int(upperbody_centroid.y)), radius=5, color=(0, 0, 255), thickness=-1)

        # Fuzzify and defuzzify values
        defuzzify_horizontal(fuzzify_horizontal(upperbody_centroid))
        defuzzify_vertical(fuzzify_vertical(upperbody_centroid))

        # Make frame smaller for preview
        if PREVIEW_ENABLED:
            cv2.imshow("Preview", cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR))
        
            # Exit if Esc key is pressed
            keypressed = cv2.waitKey(PREVIEW_REFRESH_RATE_MS) & 0xff
            if keypressed == 27:
                break

# Close the video stream
camera.close()

# Close the preview window
cv2.destroyAllWindows()

# Detach from servos
pan_servo.detach()
tilt_servo.detach()
