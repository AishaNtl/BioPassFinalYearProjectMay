# main execution file
import cv2
import dlib
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance
import os

# For the shape predictor file
PREDICTOR_PATH = r"C:\Users\Aisha\OneDrive - National College of Ireland\FinalYear\BioPassTestCodes\shape_predictor_68_face_landmarks.dat"

SOUNDS_DIR = os.path.join(r"C:\Users\Aisha\OneDrive - National College of Ireland\FinalYear\BioPassTestCodes", "sounds")
ding_sound = os.path.join(SOUNDS_DIR, "ding_sound.wav")
buzzer_sound = os.path.join(SOUNDS_DIR, "buzzer_sound.wav")

# __file__ is good recommended for portability, but it is not available in all environments.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat")
SOUNDS_DIR = os.path.join(SCRIPT_DIR, "sounds")
ding_sound = os.path.join(SOUNDS_DIR, "ding_sound.wav")
buzzer_sound = os.path.join(SOUNDS_DIR, "buzzer_sound.wav")

def verify_files():
    required_files = {
        "Shape predictor": PREDICTOR_PATH,
        "Ding sound": ding_sound,
        "Buzzer sound": buzzer_sound
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} at: {path}")
        print(f"Found {name} at: {path}")

verify_files()  # Add this before video capture initialization to make sure files are where they should be and will perform as expected

# Initialize webcam
# video = cv2.VideoCapture(0)
# Replacing current video initialization to handle errors and ensure the webcam is available before proceeding:
try:
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam")
except Exception as e:
    print(f"Webcam error: {str(e)}")
    exit(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # Now using the defined path

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(r"C:\Users\Aisha\OneDrive - National College of Ireland\FinalYear\BioPassTestCodes\shape_predictor_68_face_landmarks.dat")

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Updated expressions
expressions = ["Raised Eyebrows", "Shocked", "Happy", "Pout", "Frown", "Wink"]

# trying to find "universal" threshholds for expressions has been one of the most difficult parts of this project and has been extremely finnicky, so I have decided to use the neutral face calibration method to find the thresholds for each expression. This will allow for a more accurate and personalized detection of expressions based on the user's unique facial features.
EXPRESSION_THRESHOLDS = {
'''
    "eyebrow_raise": 5.0,    # Increased from 3.0 to require more obvious raises
    "shocked": 8.0,          # Increased from 5.0
    "happy_width": 15.0,     # Increased from 10.0
    "happy_height": -3.0,    # More pronounced smile
    "wink_ear": 0.15         # More sensitive wink detection (lower EAR)

'''
#testing stricter thresholds for expressions
# I have found that the thresholds for the expressions are too lenient and need to be stricter to ensure that the user is performing the expression correctly. I have also found that the expressions are not being detected correctly and need to be more pronounced in order to be detected. I have also found that the wink detection is too sensitive and needs to be less sensitive in order to avoid false positives.
    "eyebrow_raise": 8.0,    # Was 5.0 - requires more pronounced raise
    "shocked": 12.0,         # Was 8.0 - needs wider open mouth
    "happy_width": 20.0,     # Was 15.0 - needs bigger smile
    "happy_height": -5.0,    # Was -3.0 - more pronounced smile
    "wink_ear": 0.10         # Was 0.15 - needs more complete eye closure

}


# house all audio effects in a 'sounds' folder in my project directory
SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
ding_sound = os.path.join(SOUNDS_DIR, "ding_sound.wav")
buzzer_sound = os.path.join(SOUNDS_DIR, "buzzer_sound.wav")

# modified play_sound function to handle missing files and errors more gracefully
def play_sound(sound_file):
    try:
        if not os.path.exists(sound_file):
            print(f"Sound file missing: {sound_file}")
            return False
            
        wave_obj = sa.WaveObject.from_wave_file(sound_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        return True
    except Exception as e:
        print(f"Sound error: {str(e)}")
        return False

# Countdown function
def countdown(message="Hold Still ! Starting in,"):
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"{message} {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("BioPass Authentication", frame)
        cv2.waitKey(1000)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Function to detect facial expressions
def detect_expression(landmarks):
    global neutral_eyebrow_height, neutral_mouth_height, neutral_mouth_width, neutral_ear

    # Measure facial features
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                      (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate eyebrow height
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    # Calculate EAR for wink detection
    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS]
    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS]
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    #print out the measurements for debugging
    print(
    f"Eyebrow: {avg_eyebrow_height:.1f} (change{avg_eyebrow_height - neutral_eyebrow_height:+.1f}), "
    f"Mouth: {mouth_width:.1f}x{mouth_height:.1f} (changeW{mouth_width - neutral_mouth_width:+.1f}, changeH{mouth_height - neutral_mouth_height:+.1f}), "
    f"EAR: {left_ear:.2f}/{right_ear:.2f}"
)

    # Expression detection with dynamic thresholds

    #check for complex expressions first, then simple ones
    if mouth_width > neutral_mouth_width + 10 and mouth_height < neutral_mouth_height - 2:  # Happy
        return "Happy"
    elif mouth_height < neutral_mouth_height - 2 and mouth_width < neutral_mouth_width - 5:  # Pout
        return "Pout"
    elif left_ear < neutral_ear - 0.2 or right_ear < neutral_ear - 0.2:  # Wink
        return "Wink" 
    
        #simpler expressions are checked last as they are more likely to be detected by the model
    elif mouth_height > neutral_mouth_height + 5:  # Shocked
        return "Shocked"
    elif avg_eyebrow_height > neutral_eyebrow_height + 3:  # raised brows
        return "Raised Eyebrows"
    elif avg_eyebrow_height < neutral_eyebrow_height - 3:  # Frown
        return "Frown"
    else:
        return "Neutral"

# Function to draw facial landmarks
def draw_landmarks(frame, landmarks):
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# New Function to draw a progress bar so a user can be more aware of how close they are to completing the expression
def draw_metrics(frame, landmarks, neutral_vals):
    """Draw real-time measurements vs neutral baselines"""
    # Get current measurements
    mouth_h = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                (landmarks.part(66).x, landmarks.part(66).y))
    mouth_w = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                               (landmarks.part(54).x, landmarks.part(54).y))
    eyebrow = np.mean([abs(landmarks.part(i).y - landmarks.part(27).y) 
                      for i in [21, 22]])
    
    left_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                             for i in LEFT_EYE_LANDMARKS])
    right_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                              for i in RIGHT_EYE_LANDMARKS])
    
    # Display metrics
    y_offset = 30
    for name, current, neutral in [
        ("Eyebrow", eyebrow, neutral_vals['eyebrow']),
        ("Mouth H", mouth_h, neutral_vals['mouth_h']),
        ("Mouth W", mouth_w, neutral_vals['mouth_w']),
        ("Left EAR", left_ear, neutral_vals['ear']),
        ("Right EAR", right_ear, neutral_vals['ear'])
    ]:
        diff = current - neutral
        color = (0, 255, 0) if abs(diff) > 0 else (255, 255, 255)
        cv2.putText(frame, f"{name}: {current:.1f} ({diff:+.1f})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
    
    return frame

def draw_progress_bars(frame, landmarks, neutral_vals, target_expression):
    """Visual feedback for expression completion"""
    # Define thresholds for each expression
    THRESHOLDS = {
        "Raised Eyebrows": ("eyebrow", +3.0),
        "Frown": ("eyebrow", -3.0),
        "Shocked": ("mouth_h", +5.0),
        "Happy": ("mouth_w", +10.0),
        "Pout": ("mouth_h", -2.0),
        "Wink": ("ear", -0.2)
    }
    
    if target_expression not in THRESHOLDS:
        return
    
    metric, target_diff = THRESHOLDS[target_expression]
    current_val = {
        "eyebrow": np.mean([abs(landmarks.part(i).y - landmarks.part(27).y) 
                   for i in [21, 22]]),
        "mouth_h": distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                   (landmarks.part(66).x, landmarks.part(66).y)),
        "mouth_w": distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                  (landmarks.part(54).x, landmarks.part(54).y)),
        "ear": min(calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                                for i in LEFT_EYE_LANDMARKS]),
                  calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in RIGHT_EYE_LANDMARKS]))
    }[metric]
    
    neutral = neutral_vals[metric]
    progress = min(abs(current_val - neutral) / abs(target_diff), 1.0)
    
    # Draw progress bar
    bar_width = 200
    cv2.rectangle(frame, (50, 400), (50 + int(bar_width * progress), 420), 
                 (0, 255, 0), -1)
    cv2.rectangle(frame, (50, 400), (50 + bar_width, 420), (255, 255, 255), 2)
    cv2.putText(frame, f"{target_expression}: {progress*100:.0f}%", 
               (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
 
# Calibrate neutral face
def calibrate_neutral_face():
    global neutral_eyebrow_height, neutral_mouth_height, neutral_mouth_width, neutral_ear

    print("Calibrating neutral face... Please stay still.")
    time.sleep(2)

    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame for calibration.")
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        print("No face detected during calibration.")
        return False

    landmarks = predictor(gray, faces[0])

    # Calculate neutral measurements
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    neutral_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    neutral_mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                              (landmarks.part(66).x, landmarks.part(66).y))
    neutral_mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                             (landmarks.part(54).x, landmarks.part(54).y))

    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS]
    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS]
    neutral_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

    print("Calibration complete!")
    return True

# Blink detection
def blink_detection(blinks_required):
    blink_count = 0
    while blink_count < blinks_required:
        ret, frame = video.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS]
            
            avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            
            if avg_ear < 0.2:
                blink_count += 1
                time.sleep(0.5)  # Prevent double counting
            
        cv2.putText(frame, f"Blinks: {blink_count}/{blinks_required}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("BioPass Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    return True

# ====== MODIFY EXISTING FUNCTIONS ======
def detect_expression_with_timer(expression, time_limit=30):
    start_time = time.time()
    neutral_vals = {
        'eyebrow': neutral_eyebrow_height,
        'mouth_h': neutral_mouth_height,
        'mouth_w': neutral_mouth_width,
        'ear': neutral_ear
    }

    # Minimum duration tracking for the expression to be detected
    expression_start_time = None

    while (time.time() - start_time) < time_limit:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            landmarks = predictor(gray, face)
            
            # Visual feedback
            frame = draw_metrics(frame, landmarks, neutral_vals)
            draw_progress_bars(frame, landmarks, neutral_vals, expression)
            draw_landmarks(frame, landmarks)
            
            detected_expression = detect_expression(landmarks)
            
            # Display current attempt
            cv2.putText(frame, f"Target: {expression}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Attempting: {detected_expression}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Time left: {int(time_limit - (time.time() - start_time))}s", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if detected_expression == expression:
                play_sound(ding_sound)
                return True

        cv2.imshow("BioPass Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return False

# Main authentication flow
def start_authentication():
    # Initial countdown
    countdown()
    
    # Calibrate neutral face
    if not calibrate_neutral_face():
        return False

    # Blink detection
    blinks_required = random.randint(1, 3)
    print(f"Please blink {blinks_required} times!")
    
    if not blink_detection(blinks_required):
        print("Blink detection failed!")
        play_sound(buzzer_sound)
        return False

    # Define available expressions (must EXACTLY match what detect_expression() returns)
    available_expressions = ["Raised Eyebrows", "Frown", "Shocked", "Happy", "Pout", "Wink"]
    
    # Randomly select an expression
    selected_expression = random.choice(available_expressions)
    print(f"Perform this expression: {selected_expression}")
    

    # Clear the screen and show new instruction
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f"Perform: {selected_expression}", (150, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Starting in 3 seconds...", (150, 250), 
                #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
   # cv2.putText(frame, "Hold for 1-2 seconds", (150, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.imshow("BioPass Authentication", frame)
    cv2.waitKey(3000)  # 3 second countdown to read instructions
        
        
    # Run detection with 30-second timer for a single attempt
    success = detect_expression_with_timer(selected_expression, time_limit=30)
        
    # result feedback
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    if success:
        cv2.putText(frame, "SUCCESS!", (200, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Detected: {selected_expression}", (100, 250), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        play_sound(ding_sound)
    else:
        cv2.putText(frame, "TRY AGAIN!", (200, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"Failed: {selected_expression}", (100, 250), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        play_sound(buzzer_sound)

        cv2.imshow("BioPass Authentication", frame)
        cv2.waitKey(3000)  # display result for 3 seconds
        return success


# Run the authentication
start_authentication()

# Cleanup
video.release()
cv2.destroyAllWindows()