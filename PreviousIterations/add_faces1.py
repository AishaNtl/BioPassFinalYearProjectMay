# BioPass Facial Expression CAPTCHA Authentication System
# This script implements a facial expression-based authentication system using OpenCV and Dlib.
import cv2
import dlib
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance
import os

# ====== PATH CONFIGURATION ======
def setup_paths():
    """Centralized path configuration using the most reliable method"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    return {
        'predictor': os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat"),
        'sounds_dir': os.path.join(script_dir, "sounds"),
        'ding_sound': os.path.join(script_dir, "sounds", "ding_sound.wav"),
        'buzzer_sound': os.path.join(script_dir, "sounds", "buzzer_sound.wav"),
        'calibration_sound': os.path.join(script_dir, "sounds", "calibration_success.wav")
    }

paths = setup_paths()

# ====== FILE VERIFICATION ======
def verify_files(paths):
    """Verify all required files exist"""
    for name, path in paths.items():
        if not os.path.exists(path):
            if name != 'calibration_sound':  # Make calibration sound optional
                raise FileNotFoundError(f"Missing {name} at: {path}")
        print(f"[{'FOUND' if os.path.exists(path) else 'MISSING'}] {name.replace('_', ' ')} at: {path}")

verify_files(paths)

# ====== VIDEO INITIALIZATION ======
def init_camera():
    """Initialize and verify webcam access"""
    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise RuntimeError("Could not open webcam")
        return video
    except Exception as e:
        print(f"Webcam error: {str(e)}")
        exit(1)

video = init_camera()

# ====== DLIB INITIALIZATION ======
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(paths['predictor'])

# ====== CONSTANTS ======
# Define facial landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Expression options
EXPRESSIONS = ["Raised Eyebrows", "Shocked", "Happy", "Pout", "Frown", "Wink"]

# Global variables for dynamic thresholds
neutral_eyebrow_height = 0
neutral_mouth_height = 0
neutral_mouth_width = 0
neutral_ear = 0

# ====== UTILITY FUNCTIONS ======
def calculate_ear(eye_points):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def play_sound(sound_file):
    """Play sound with error handling"""
    try:
        if os.path.exists(sound_file):
            wave_obj = sa.WaveObject.from_wave_file(sound_file)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            return True
        return False
    except Exception as e:
        print(f"Sound error: {str(e)}")
        return False

def draw_landmarks(frame, landmarks):
    """Draw facial landmarks on frame"""
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# ====== CORE FUNCTIONALITY ======
def calibrate_neutral_face():
    global neutral_eyebrow_height, neutral_mouth_height, neutral_mouth_width, neutral_ear
    
    print("Calibrating neutral face... Please maintain a neutral expression.")
    time.sleep(1)  # Give user time to prepare
    
    # Take multiple samples for better accuracy
    samples = []
    for _ in range(5):  # Take 5 samples
        ret, frame = video.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        
        if len(faces) == 1:  # Only use frames with exactly one face
            landmarks = predictor(gray, faces[0])
            
            # Get measurements
            samples.append({
                'eyebrow': np.mean([abs(landmarks.part(i).y - landmarks.part(27).y) 
                                   for i in [21, 22]]),
                'mouth_h': distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                            (landmarks.part(66).x, landmarks.part(66).y)),
                'mouth_w': distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                            (landmarks.part(54).x, landmarks.part(54).y)),
                'ear': np.mean([
                    calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in LEFT_EYE_LANDMARKS]),
                    calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in RIGHT_EYE_LANDMARKS])
                ])
            })
            time.sleep(0.3)  # Brief pause between samples
    
    if not samples:
        print("Calibration failed - no valid samples collected")
        return False
    
    # Calculate median values (more robust to outliers)
    neutral_eyebrow_height = np.median([s['eyebrow'] for s in samples])
    neutral_mouth_height = np.median([s['mouth_h'] for s in samples])
    neutral_mouth_width = np.median([s['mouth_w'] for s in samples])
    neutral_ear = np.median([s['ear'] for s in samples])
    
    # Play calibration complete sound
    if not play_sound(paths.get('calibration_sound', paths['ding_sound'])):
        print("Calibration complete sound not available")
    
    print(f"Calibration complete! Baseline values:\n"
          f"Eyebrow: {neutral_eyebrow_height:.2f}px\n"
          f"Mouth Height: {neutral_mouth_height:.2f}px\n"
          f"Mouth Width: {neutral_mouth_width:.2f}px\n"
          f"EAR: {neutral_ear:.2f}")
    return True

def detect_expression(landmarks):
    """Detect facial expression based on landmark positions"""
    # Current measurements
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                    (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                   (landmarks.part(54).x, landmarks.part(54).y))
    eyebrow_height = np.mean([abs(landmarks.part(i).y - landmarks.part(27).y) 
                            for i in [21, 22]])
    
    # Calculate EAR for both eyes
    left_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                             for i in LEFT_EYE_LANDMARKS])
    right_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                              for i in RIGHT_EYE_LANDMARKS])
    
    # Relative changes from neutral
    eyebrow_change = (eyebrow_height - neutral_eyebrow_height) / neutral_eyebrow_height
    mouth_h_change = (mouth_height - neutral_mouth_height) / neutral_mouth_height
    mouth_w_change = (mouth_width - neutral_mouth_width) / neutral_mouth_width
    
    # Dynamic thresholds (adjust these based on your testing)
    if eyebrow_change > 0.10:    return "Raised Eyebrows"
    elif eyebrow_change < -0.15: return "Frown"
    elif mouth_h_change > 0.25:  return "Shocked"
    elif (mouth_w_change > 0.20 and 
          mouth_h_change > -0.10): return "Happy"
    elif (mouth_h_change < -0.20 and 
          mouth_w_change < -0.15): return "Pout"
    elif (left_ear < neutral_ear*0.6 or 
          right_ear < neutral_ear*0.6): return "Wink"
    
    return "Neutral"

def detect_expression_with_timer(target_expression):
    """Detect if user performs target expression within time limit"""
    print(f"Please perform: {target_expression}")
    start_time = time.time()
    expression_held_for = 0
    required_hold_time = 1.5  # Seconds expression must be held
    
    while (time.time() - start_time) < 30:  # 15 second timeout
        ret, frame = video.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        
        for face in faces:
            landmarks = predictor(gray, face)
            current_expression = detect_expression(landmarks)
            
            # Visual feedback
            color = (0, 255, 0) if current_expression == target_expression else (0, 0, 255)
            cv2.putText(frame, f"Target: {target_expression}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Current: {current_expression}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Time: {int(15 - (time.time() - start_time))}s",
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if current_expression == target_expression:
                expression_held_for += 0.1  # Approximate based on loop timing
                if expression_held_for >= required_hold_time:
                    play_sound(paths['ding_sound'])
                    return True
            else:
                expression_held_for = 0
            
        cv2.imshow("BioPass Authentication", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    play_sound(paths['buzzer_sound'])
    return False

def blink_detection(blinks_required):
    """Detect specified number of blinks"""
    blink_count = 0
    last_blink_time = 0
    
    while blink_count < blinks_required:
        ret, frame = video.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        
        for face in faces:
            landmarks = predictor(gray, face)
            left_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                                    for i in LEFT_EYE_LANDMARKS])
            right_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) 
                                     for i in RIGHT_EYE_LANDMARKS])
            
            # Consider it a blink if either eye is closed
            if left_ear < 0.2 or right_ear < 0.2:
                if time.time() - last_blink_time > 0.5:  # Debounce
                    blink_count += 1
                    last_blink_time = time.time()
                    print(f"Blink detected! ({blink_count}/{blinks_required})")
            
        # Visual feedback
        cv2.putText(frame, f"Blinks: {blink_count}/{blinks_required}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("BioPass Authentication", frame)
        
        if cv2.waitKey(1) == ord('q'):
            return False
    
    return True

def start_authentication():
    """Main authentication flow"""
    # Initial countdown
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Starting in {i}...", (200, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("BioPass Authentication", frame)
        cv2.waitKey(1000)
    
    # Calibrate neutral face
    if not calibrate_neutral_face():
        play_sound(paths['buzzer_sound'])
        return
    
    # Blink detection
    blinks_required = random.randint(1, 3)
    print(f"Please blink {blinks_required} times!")
    if not blink_detection(blinks_required):
        print("Blink detection failed!")
        play_sound(paths['buzzer_sound'])
        return
    
    # Expression detection
    random.shuffle(EXPRESSIONS)
    for i, expression in enumerate(EXPRESSIONS[:2]):  # Test 2 random expressions
        if i > 0:  # Add countdown between expressions (skip first)
            for j in range(3, 0, -1):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Next in {j}...", (200, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("BioPass Authentication", frame)
                cv2.waitKey(1000)
        
        print(f"Perform: {expression}")
        if not detect_expression_with_timer(expression):
            print(f"Failed to detect: {expression}")
            return
    
    # Success
    print("Authentication successful!")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Authentication Successful!", (100, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("BioPass Authentication", frame)
    cv2.waitKey(3000)

# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    try:
        start_authentication()
    except KeyboardInterrupt:
        print("\nAuthentication cancelled by user")
    finally:
        video.release()
        cv2.destroyAllWindows()