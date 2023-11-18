# Jaron Anderson
# SSE 676
# Fuzzy algorithm to adjust speed of two motors on a two-axis camera mount, 
# one for pan and one for tilt, to track a subject and keep them roughly in the center of the frame.

# IMPORTANT: cv2 defaults to BGR color space
import cv2
import numpy as np

PREVIEW_REFRESH_RATE_MS = 20

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
			# Bring pan motor to a stop
			print("Stop moving in the x!")
	elif membership_values[0] > 0.5:
		# Accelerate panning motor in + direction
		print("Accelerate in the +x!")
	else:
		# Accelerate panning motor in - direction
		print("Accelerate in the -x!")

def defuzzify_vertical(membership_values):
	# membership_values[0] = exiting_top
	# membership_values[1] = not_exiting
	# membership_values[2] = exiting_bottom

	if membership_values[1] > 0.5:
		if membership_values[1] > 0.8:
			# Bring tilt motor to a stop
			print("Stop moving in the y!")
	elif membership_values[0] > 0.5:
		# Accelerate tilting motor in - direction
		print("Accelerate in the -y!")
	else:
		# Accelerate tilting motor in + direction
		print("Accelerate in the +y!")

################## END DEFUZZIFICATION FUNCTIONS ####################

# Load trained XML classifier for detecting upper body
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Create object of the Centroid class to track the upper body of subject
# Initialize to the center of the frame
upperbody_centroid = Centroid(1920 / 2, 1080 / 2)

# Establish connection to video, 
video_stream = cv2.VideoCapture(1)
if not video_stream.isOpened():
	print("Cannot open camera. Exiting...")
	exit()
video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_count = 0
dropped_frames = 0

# Loop forever
while(True):
	# Capture a frame and process it
	captured, frame = video_stream.read()
	frame_count += 1
	if not captured:
		dropped_frames += 1
		print("Frame dropped.")
		continue
	
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect upper bodies in frame
	upperbodies = upperbody_cascade.detectMultiScale(gray)

	for (x,y,w,h) in upperbodies: 
        # Draw rectangle on detected upperbodies
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
		upperbody_centroid.x = x + (w / 2)
		upperbody_centroid.y = y + (h / 2)

	# Fuzzify and defuzzify values
	defuzzify_horizontal(fuzzify_horizontal(upperbody_centroid))
	defuzzify_vertical(fuzzify_vertical(upperbody_centroid))

	# Make frame smaller for preview
	cv2.imshow("Preview", cv2.resize(frame, (1280, 720)))
	
	# Exit if Esc key is pressed
	keypressed = cv2.waitKey(PREVIEW_REFRESH_RATE_MS) & 0xff
	if keypressed == 27:
		break

# Close the video stream
video_stream.release()

# Close the preview window
cv2.destroyAllWindows()