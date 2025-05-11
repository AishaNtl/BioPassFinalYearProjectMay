'''
import cv2
import numpy as np

video = cv2.VideoCapture(0) #using 0 uses the inbuilt camera in this case my laptop camer while parsing 1 uses an external camera
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml') #file path to face training data


while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:    
        cv2.rectangle(frame, (x,y), (x+w, y+h), (224,33,138), 4)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
'''

'''
#code without blink detect
import cv2
import dlib
import os
import numpy as np

# Initialize webcam
video = cv2.VideoCapture(0)

# Load dlib's face detector (HOG + SVM-based) and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure file is available

# Create output folder for processed images
output_folder = "processed_faces"
os.makedirs(output_folder, exist_ok=True)

# Function to preprocess image (grayscale, resize, normalize)
def preprocess_image(image, size=(150, 150)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, size)  # Resize
    normalized = cv2.equalizeHist(resized)  # Normalize brightness/contrast
    return normalized

# Function to apply data augmentation (flip, rotate, brightness)
def augment_image(image):
    flipped = cv2.flip(image, 1)  # Flip horizontally
    brightness = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Increase brightness
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees
    return [image, flipped, brightness, rotated]

# Function to detect faces and landmarks
def detect_face_and_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces
    landmarks_list = []

    for face in faces:
        landmarks = predictor(gray, face)  # Get facial landmarks
        landmarks_list.append(landmarks)

    return faces, landmarks_list

frame_count = 0  # Counter for saving frames

while True:
    # Capture frame
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # Resize for faster processing

    # Detect faces and landmarks
    faces, landmarks_list = detect_face_and_landmarks(frame)

    for face, landmarks in zip(faces, landmarks_list):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (147, 20, 255), 2)  # Pink bounding box

        # Extract face region
        face_img = frame[y:y + h, x:x + w]

        # Preprocess and augment face
        processed_face = preprocess_image(face_img)
        augmented_faces = augment_image(processed_face)

        # Save processed and augmented images
        for i, aug_img in enumerate(augmented_faces):
            filename = f"{output_folder}/face_{frame_count}_{i}.jpg"
            cv2.imwrite(filename, aug_img)

        frame_count += 1  # Increment frame count

        # Draw facial landmarks
        for i in range(68):
            lx, ly = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)  # Green dots for landmarks

    # Display frame
    cv2.imshow("Face & Landmark Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()

print("Process completed! Preprocessed & augmented images saved.")
'''

'''
#code with blink detect
import cv2
import dlib
from scipy.spatial import distance

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices for eyes in the 68-point landmark model
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < 0.2:  # Blink detected if eyes close momentarily
            cv2.putText(frame, "Blink Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

'''
import cv2
import dlib
import os
import numpy as np
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

# Load dlib's face detector (HOG + SVM-based) and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure file is available

# Create output folder for processed images
output_folder = "processed_faces"
os.makedirs(output_folder, exist_ok=True)

# Define eye landmark indexes for blink detection
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))

# Function to preprocess image (grayscale, resize, normalize)
def preprocess_image(image, size=(150, 150)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, size)  # Resize
    normalized = cv2.equalizeHist(resized)  # Normalize brightness/contrast
    return normalized

# Function to apply data augmentation (flip, rotate, brightness)
def augment_image(image):
    flipped = cv2.flip(image, 1)  # Flip horizontally
    brightness = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Increase brightness
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees
    return [image, flipped, brightness, rotated]

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect faces, landmarks, and blinks
def detect_face_and_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces
    landmarks_list = []
    blink_status = False

    for face in faces:
        landmarks = predictor(gray, face)  # Get facial landmarks
        landmarks_list.append(landmarks)

        # Extract eye landmarks
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS]

        # Compute eye aspect ratio (EAR)
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if EAR is below the blink threshold
        if avg_ear < 0.2:
            blink_status = True

    return faces, landmarks_list, blink_status

frame_count = 0  # Counter for saving frames
blink_counter = 0  # Counter for blink detection
blink_threshold = 3  # Number of consecutive frames to confirm a blink

while True:
    # Capture frame
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # Resize for faster processing

    # Detect faces, landmarks, and blinks
    faces, landmarks_list, blink_detected = detect_face_and_landmarks(frame)

    for face, landmarks in zip(faces, landmarks_list):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (147, 20, 255), 2)  # Pink bounding box

        # Extract face region
        face_img = frame[y:y + h, x:x + w]

        # Preprocess and augment face
        processed_face = preprocess_image(face_img)
        augmented_faces = augment_image(processed_face)

        # Save processed and augmented images
        for i, aug_img in enumerate(augmented_faces):
            filename = f"{output_folder}/face_{frame_count}_{i}.jpg"
            cv2.imwrite(filename, aug_img)

        frame_count += 1  # Increment frame count

        # Draw facial landmarks
        for i in range(68):
            lx, ly = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)  # Green dots for landmarks

    # Blink detection message
    if blink_detected:
        blink_counter += 1
        if blink_counter >= blink_threshold:
            cv2.putText(frame, "BLINK DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        blink_counter = 0  # Reset counter if no blink detected

    # Display frame
    cv2.imshow("Face, Landmark, and Blink Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()

print("Process completed! Preprocessed & augmented images saved.")
'''

'''
import cv2
import dlib
import os
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

output_folder = "processed_faces"
os.makedirs(output_folder, exist_ok=True)

# Define eye landmark indexes for blink detection
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Facial expression prompts
expressions = ["Neutral", "Smile", "Surprised", "Frown"]
random.shuffle(expressions)

# Load sound for feedback
ding_sound = "ding-36029.wav"  # Ensure'ding-36029.wav is in' file in the same directory

def play_sound():
    wave_obj = sa.WaveObject.from_wave_file(ding_sound)
    wave_obj.play()

# Countdown function
def countdown():
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Starting in {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Liveness Detection", frame)
        cv2.waitKey(1000)

# Function to preprocess image
def preprocess_image(image, size=(150, 150)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    normalized = cv2.equalizeHist(resized)
    return normalized

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Function to detect expressions
def detect_expression(landmarks, frame):
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                      (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))
    
    left_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS])
    right_ear = calculate_ear([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS])
    avg_ear = (left_ear + right_ear) / 2.0
    
    if mouth_height > 20:  # Detect Surprised
        return "Surprised"
    elif mouth_width > 60 and mouth_height < 15:  # Detect Smile
        return "Smile"
    elif avg_ear < 0.2:  # Detect Frown (simulated by squinting)
        return "Frown"
    else:
        return "Neutral"

countdown()  # Start countdown before expressions

for expression in expressions:
    success = False
    start_time = time.time()
    
    while time.time() - start_time < 5:  # Give user 5 seconds to perform expression
        ret, frame = video.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            detected_expression = detect_expression(landmarks, frame)
            
            if detected_expression == expression:
                success = True
                play_sound()
                break
            
            # Draw bounding box
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (147, 20, 255), 2)
            
            # Draw landmarks
            for i in range(68):
                lx, ly = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
            
            cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if not success:
        print(f"Failed to detect {expression}, retrying...")
        break

video.release()
cv2.destroyAllWindows()
print("Liveness check complete!")
'''
'''
import cv2
import dlib
import os
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

output_folder = "processed_faces"
os.makedirs(output_folder, exist_ok=True)

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Updated expressions
expressions = ["Raised Eyebrows", "Shocked", "Happy", "Pout"]

# Load sound for feedback
ding_sound = "ding-36029.wav"  
buzzer_sound = "buzzer.wav"  

def play_sound(sound_file):
    wave_obj = sa.WaveObject.from_wave_file(sound_file)
    wave_obj.play()

# Countdown before starting
def countdown():
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Starting in {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Liveness Detection", frame)
        cv2.waitKey(1000)

# Function to calculate Eye Aspect Ratio (EAR) for eye-based expressions
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Function to detect facial expressions
def detect_expression(landmarks):
    # Measure mouth & eyebrow positions
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                      (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate eyebrow height (fixing the NameError)
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    # Expression detection logic
    if avg_eyebrow_height > 10:  # Adjusted for better Raised Eyebrow detection
        return "Raised Eyebrows"
    elif mouth_height > 20:  # Shocked (mouth open wide)
        return "Shocked"
    elif mouth_width > 60 and mouth_height < 15:  # Happy (smile)
        return "Happy"
    elif mouth_height < 10 and mouth_width < 40:  # Pout (pushed-out lips)
        return "Pout"
    else:
        return "Neutral"


# Blink detection function
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
                time.sleep(0.5)  # Prevents double counting
            
        cv2.putText(frame, f"Blinks: {blink_count}/{blinks_required}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    return True

# Main function to run the authentication process
def start_authentication():
    countdown()
    
    # Random number of blinks required
    blinks_required = random.randint(1, 3)
    print(f"Please blink {blinks_required} times!")
    
    if not blink_detection(blinks_required):
        print("Blink detection failed! Please try again.")
        return
    
    # Shuffle and start expression detection
    random.shuffle(expressions)

    for expression in expressions:
        success = False
        
        while not success:
            # = time.time()
            
            #while time.time() - start_time < 5:  # Give 5 seconds per attempt
                ret, frame = video.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 1)
                
                for face in faces:
                    landmarks = predictor(gray, face)
                    detected_expression = detect_expression(landmarks)
                    
                    if detected_expression == expression:
                        success = True
                        play_sound(ding_sound)  # Play 'ding' sound on success
                        time.sleep(1) #small delay before moving to next expression because program kept breaking after first expression prompt
                    
                    # Draw bounding box around face
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (147, 20, 255), 2)
                    padding = 20  # Adjust this value as needed
                    cv2.rectangle(frame, (x - padding, y - padding), 
                                  (x + w + padding, y + h + padding), 
                                  (147, 20, 255), 2)

                    # Draw landmarks
                    for i in range(68):
                        lx, ly = landmarks.part(i).x, landmarks.part(i).y
                        cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
                    
                    cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow("Facial Expression Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            
        if not success:
                play_sound(buzzer_sound)  # Play 'buzzer' sound on failure
                print(f"Failed to detect {expression}, retrying...")

    # Success message after all expressions are correctly performed
    print("All expressions successfully detected!")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "BioPass Welcomes You!", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow("Authentication Complete", frame)
    cv2.waitKey(3000)

# Start authentication process
start_authentication()

# Release resources
video.release()
cv2.destroyAllWindows()
'''

'''
import cv2
import dlib
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance
import threading

# Initialize webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Only test for the "Happy" expression
expressions = ["Happy"]

# Load sound for feedback
ding_sound = "ding-36029.wav"
buzzer_sound = "buzzer.wav"

# Constants
MAX_BLINK_ATTEMPTS = 3
MAX_EXPRESSION_ATTEMPTS = 2
LOCKOUT_TIME = 30  # Seconds
SESSION_TIMEOUT = 60  # Seconds

# Function to play sound asynchronously
def play_sound(sound_file):
    def play():
        wave_obj = sa.WaveObject.from_wave_file(sound_file)
        wave_obj.play()
    threading.Thread(target=play).start()

# Countdown before starting
def countdown():
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Starting in {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("BioPass", frame)
        cv2.waitKey(1000)

# Function to calculate Eye Aspect Ratio (EAR) for eye-based expressions
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Function to detect facial expressions
def detect_expression(landmarks):
    # Measure mouth & eyebrow positions
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                      (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate eyebrow height
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    # Expression detection logic (only for "Happy")
    if mouth_width > 60 and mouth_height < 15:  # Happy (smile)
        return "Happy"
    else:
        return "Neutral"

# Blink detection function
def blink_detection(blinks_required):
    blink_count = 0
    attempts = 0

    while blink_count < blinks_required and attempts < MAX_BLINK_ATTEMPTS:
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
                play_sound(ding_sound)  # Play sound on successful blink
                time.sleep(0.5)  # Prevents double counting

        cv2.putText(frame, f"Blinks: {blink_count}/{blinks_required}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("BioPass", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        attempts += 1

    if attempts >= MAX_BLINK_ATTEMPTS:
        print("Too many failed attempts. Please try again later.")
        time.sleep(LOCKOUT_TIME)
        return False

    return True

# Main function to run the authentication process
def start_authentication():
    try:
        # Display GDPR notice
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Your data will not be stored.", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press any key to continue...", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("BioPass", frame)
        cv2.waitKey(0)

        countdown()

        # Random number of blinks required
        blinks_required = random.randint(1, 3)
        print(f"Please blink {blinks_required} times!")

        if not blink_detection(blinks_required):
            print("Blink detection failed! Please try again.")
            return

        # Only test for the "Happy" expression
        expression = "Happy"
        success = False
        attempts = 0

        while not success and attempts < MAX_EXPRESSION_ATTEMPTS:
            ret, frame = video.read()
            if not ret:
                print("Failed to read frame!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)

            for face in faces:
                landmarks = predictor(gray, face)
                detected_expression = detect_expression(landmarks)

                if detected_expression == expression:
                    success = True
                    play_sound(ding_sound)  # Play 'ding' sound on success
                    time.sleep(1)  # Small delay before moving to next expression

                # Draw bounding box around face
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                padding = 20
                cv2.rectangle(frame, (x - padding, y - padding),
                              (x + w + padding, y + h + padding),
                              (147, 20, 255), 2)

                # Draw landmarks
                for i in range(68):
                    lx, ly = landmarks.part(i).x, landmarks.part(i).y
                    cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)

                cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("BioPass", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

            attempts += 1

        if not success:
            play_sound(buzzer_sound)  # Play 'buzzer' sound on failure
            print(f"Failed to detect {expression}, retrying...")

        # Success message after the expression is correctly performed
        print("Expression successfully detected!")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "BioPass Welcomes You!", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow("BioPass", frame)
        cv2.waitKey(3000)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        video.release()
        cv2.destroyAllWindows()

# Start authentication process
start_authentication()
'''

'''
import cv2
import dlib
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance
import threading

# Initialize webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Load sound for feedback
ding_sound = "ding-36029.wav"
buzzer_sound = "buzzer.wav"

# Constants
MAX_BLINK_ATTEMPTS = 3
MAX_EXPRESSION_ATTEMPTS = 2
LOCKOUT_TIME = 30  # Seconds

'''
'''
import cv2
import dlib
import os
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

output_folder = "processed_faces"
os.makedirs(output_folder, exist_ok=True)

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Updated expressions
expressions = ["Raised Eyebrows", "Shocked", "Happy", "Pout"]

# Load sound for feedback
ding_sound = "ding-36029.wav"  
buzzer_sound = "wrong-38598.wav"  

def play_sound(sound_file):
    wave_obj = sa.WaveObject.from_wave_file(sound_file)
    wave_obj.play()

# Countdown before starting
def countdown():
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Starting in {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Liveness Detection", frame)
        cv2.waitKey(1000)

# Function to calculate Eye Aspect Ratio (EAR) for eye-based expressions
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Function to detect facial expressions
def detect_expression(landmarks):
    # Measure mouth & eyebrow positions
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                      (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate eyebrow height (fixing the NameError)
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    # Expression detection logic
    if avg_eyebrow_height > 10:  # Adjusted for better Raised Eyebrow detection
        return "Raised Eyebrows"
    elif mouth_height > 20:  # Shocked (mouth open wide)
        return "Shocked"
    elif mouth_width > 60 and mouth_height < 15:  # Happy (smile)
        return "Happy"
    elif mouth_height < 10 and mouth_width < 40:  # Pout (pushed-out lips)
        return "Pout"
    else:
        return "Neutral"

# Blink detection function
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
                time.sleep(0.5)  # Prevents double counting
            
        cv2.putText(frame, f"Blinks: {blink_count}/{blinks_required}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    return True

# Main function to run the authentication process
def start_authentication():
    countdown()
    
    # Random number of blinks required
    blinks_required = random.randint(1, 3)
    print(f"Please blink {blinks_required} times!")
    
    if not blink_detection(blinks_required):
        print("Blink detection failed! Please try again.")
        return
    
    # Shuffle and start expression detection
    random.shuffle(expressions)

    for expression in expressions[:2]:
        success = False
        
        while not success:
            ret, frame = video.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            
            for face in faces:
                landmarks = predictor(gray, face)
                detected_expression = detect_expression(landmarks)
                
                if detected_expression == expression:
                    success = True
                    play_sound(ding_sound)  # Play 'ding' sound on success
                    time.sleep(1)
                else:
                    play_sound(buzzer_sound)
                
                cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Facial Expression Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    print("All expressions successfully detected!")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "BioPass Welcomes You!", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow("Authentication Complete", frame)
    cv2.waitKey(3000)

# Start authentication process
start_authentication()

# Release resources
#video.release()
#cv2.destroyAllWindows()'
'''
'''
import cv2
import dlib
import os
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

output_folder = "processed_faces"
os.makedirs(output_folder, exist_ok=True)

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Updated expressions
expressions = ["Raised Eyebrows", "Shocked", "Happy", "Pout"]

# Load sound for feedback
ding_sound = "ding-36029.wav"  
buzzer_sound = "wrong-38598.wav"  

def play_sound(sound_file):
    wave_obj = sa.WaveObject.from_wave_file(sound_file)
    wave_obj.play()

# Countdown before starting
def countdown(message="Starting in"):
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"{message} {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Liveness Detection", frame)
        cv2.waitKey(1000)

# Function to calculate Eye Aspect Ratio (EAR) for eye-based expressions
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Function to detect facial expressions
def detect_expression(landmarks):
    # Measure mouth & eyebrow positions
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                      (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate eyebrow height
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    # Expression detection logic
    if avg_eyebrow_height > 10:  # Raised Eyebrows
        return "Raised Eyebrows"
    elif mouth_height > 20:  # Shocked (mouth open wide)
        return "Shocked"
    elif mouth_width > 60 and mouth_height < 15:  # Happy (smile)
        return "Happy"
    elif mouth_height < 10 and mouth_width < 40:  # Pout (pushed-out lips)
        return "Pout"
    else:
        return "Neutral"

# Blink detection function
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
                time.sleep(0.5)  # Prevents double counting
            
        cv2.putText(frame, f"Blinks: {blink_count}/{blinks_required}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    return True

# Main function to run the authentication process
def start_authentication():
    countdown()
    
    # Random number of blinks required
    blinks_required = random.randint(1, 3)
    print(f"Please blink {blinks_required} times!")
    
    if not blink_detection(blinks_required):
        print("Blink detection failed! Please try again.")
        play_sound(buzzer_sound)
        return
    
    # Shuffle and select two random expressions
    random.shuffle(expressions)
    selected_expressions = expressions[:2]

    for i, expression in enumerate(selected_expressions):
        success = False
        start_time = time.time()
        time_limit = 15  # 10 seconds to complete each expression
        buzzer_played = False  # Flag to track if buzzer sound has been played
        
        # Add a countdown before each expression (except the first one)
        if i > 0:
            countdown(message="Next expression in")
        
        while not success and (time.time() - start_time) < time_limit:
            ret, frame = video.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            
            for face in faces:
                landmarks = predictor(gray, face)
                detected_expression = detect_expression(landmarks)
                
                if detected_expression == expression:
                    success = True
                    play_sound(ding_sound)  # Play 'ding' sound on success
                    time.sleep(1)
                elif not buzzer_played:
                    play_sound(buzzer_sound)
                    buzzer_played = True  # Ensure buzzer sound is played only once
                
                cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Time left: {int(time_limit - (time.time() - start_time))}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Facial Expression Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        if not success:
            print(f"Failed to detect expression: {expression}")
            return

    print("All expressions successfully detected!")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "CAPTCHA Success!", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow("Authentication Complete", frame)
    cv2.waitKey(3000)

# Start authentication process
start_authentication()

# Release resources
video.release()
cv2.destroyAllWindows()'
'''
'''
import cv2
import dlib
import numpy as np
import time
import random
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Updated expressions
expressions = ["Raised Eyebrows", "Shocked", "Happy", "Pout"]

# Countdown function
def countdown(message="Starting in"):
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"{message} {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Expression Detection", frame)
        cv2.waitKey(1000)

# Function to detect facial expressions
def detect_expression(landmarks):
    # Measure mouth & eyebrow positions
    mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                      (landmarks.part(66).x, landmarks.part(66).y))
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate eyebrow height
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    # Expression detection logic
    if avg_eyebrow_height > 10:  # Raised Eyebrows
        return "Raised Eyebrows"
    elif mouth_height > 20:  # Shocked (mouth open wide)
        return "Shocked"
    elif mouth_width > 60 and mouth_height < 15:  # Happy (smile)
        return "Happy"
    elif mouth_height < 10 and mouth_width < 40:  # Pout (pushed-out lips)
        return "Pout"
    else:
        return "Neutral"

# Function to draw facial landmarks on the frame
def draw_landmarks(frame, landmarks):
    for i in range(68):  # There are 68 landmarks in the dlib model
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw a green dot at each landmark

# Function to handle expression detection with a timer
def detect_expression_with_timer(expression):
    start_time = time.time()
    time_limit = 10  # 10 seconds to complete the expression

    while (time.time() - start_time) < time_limit:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            landmarks = predictor(gray, face)
            detected_expression = detect_expression(landmarks)

            # Draw facial landmarks on the frame
            draw_landmarks(frame, landmarks)

            if detected_expression == expression:
                return True  # Expression detected successfully

            cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Time left: {int(time_limit - (time.time() - start_time))}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Expression Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return False  # Time limit exceeded

# Main function to test expression prompts
def test_expression_prompts():
    # Shuffle and select two random expressions
    random.shuffle(expressions)
    selected_expressions = expressions[:2]

    # First expression prompt
    print(f"First expression: {selected_expressions[0]}")
    countdown(message="First expression in")
    if detect_expression_with_timer(selected_expressions[0]):
        print("First expression detected successfully!")
    else:
        print("Failed to detect the first expression.")
        return

    # Countdown before the second expression
    countdown(message="Next expression in")

    # Second expression prompt
    print(f"Second expression: {selected_expressions[1]}")
    if detect_expression_with_timer(selected_expressions[1]):
        print("Second expression detected successfully!")
    else:
        print("Failed to detect the second expression.")
        return

    print("Both expressions detected successfully!")

# Run the test
test_expression_prompts()

# Release resources
video.release()
cv2.destroyAllWindows()'
'''
'''
import cv2
import dlib
import numpy as np
import time
import random
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Updated expressions
expressions = ["Raised Eyebrows", "Shocked", "Happy", "Pout", "Frown", "Wink"]

# Global variables for dynamic thresholds
neutral_eyebrow_height = 0
neutral_mouth_height = 0
neutral_mouth_width = 0
neutral_ear = 0  # Eye Aspect Ratio for wink detection

# Countdown function
def countdown(message="Starting in"):
    for i in range(3, 0, -1):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"{message} {i}...", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Expression Detection", frame)
        cv2.waitKey(1000)

# Function to calculate Eye Aspect Ratio (EAR) for wink detection
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Function to detect facial expressions
def detect_expression(landmarks):
    global neutral_eyebrow_height, neutral_mouth_height, neutral_mouth_width, neutral_ear

    # Measure mouth & eyebrow positions
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

    # Print debugging information
    print(f"Eyebrow Height: {avg_eyebrow_height}, Mouth Height: {mouth_height}, Mouth Width: {mouth_width}, Left EAR: {left_ear}, Right EAR: {right_ear}")

    # Expression detection logic with dynamic thresholds
    if avg_eyebrow_height > neutral_eyebrow_height + 3:  # Raised Eyebrows
        return "Raised Eyebrows"
    elif avg_eyebrow_height < neutral_eyebrow_height - 3:  # Frown
        return "Frown"
    elif mouth_height > neutral_mouth_height + 5:  # Shocked (mouth open wide)
        return "Shocked"
    elif mouth_width > neutral_mouth_width + 10 and mouth_height < neutral_mouth_height - 2:  # Happy (smile)
        return "Happy"
    elif mouth_height < neutral_mouth_height - 2 and mouth_width < neutral_mouth_width - 5:  # Pout (pushed-out lips)
        return "Pout"
    elif left_ear < neutral_ear - 0.2 or right_ear < neutral_ear - 0.2:  # Wink (one eye closed)
        return "Wink"
    else:
        return "Neutral"

# Function to draw facial landmarks on the frame
def draw_landmarks(frame, landmarks):
    for i in range(68):  # There are 68 landmarks in the dlib model
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw a green dot at each landmark

# Function to calibrate neutral face
def calibrate_neutral_face():
    global neutral_eyebrow_height, neutral_mouth_height, neutral_mouth_width, neutral_ear

    print("Calibrating neutral face... Please stay still.")
    time.sleep(2)  # Give the user time to prepare

    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame for calibration.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        print("No face detected during calibration.")
        return

    landmarks = predictor(gray, faces[0])

    # Calculate neutral eyebrow height, mouth height, and mouth width
    left_eyebrow_height = abs(landmarks.part(21).y - landmarks.part(27).y)
    right_eyebrow_height = abs(landmarks.part(22).y - landmarks.part(27).y)
    neutral_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2.0

    neutral_mouth_height = distance.euclidean((landmarks.part(62).x, landmarks.part(62).y),
                                              (landmarks.part(66).x, landmarks.part(66).y))
    neutral_mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                             (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate neutral EAR
    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS]
    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS]
    neutral_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

    print("Calibration complete!")
    print(f"Neutral Eyebrow Height: {neutral_eyebrow_height}, Neutral Mouth Height: {neutral_mouth_height}, Neutral Mouth Width: {neutral_mouth_width}, Neutral EAR: {neutral_ear}")

# Function to handle expression detection with a timer
def detect_expression_with_timer(expression):
    start_time = time.time()
    time_limit = 10  # 10 seconds to complete the expression

    while (time.time() - start_time) < time_limit:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            landmarks = predictor(gray, face)
            detected_expression = detect_expression(landmarks)

            # Draw facial landmarks on the frame
            draw_landmarks(frame, landmarks)

            if detected_expression == expression:
                return True  # Expression detected successfully

            cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Time left: {int(time_limit - (time.time() - start_time))}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Expression Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return False  # Time limit exceeded

# Main function to test expression prompts
def test_expression_prompts():
    # Calibrate neutral face
    calibrate_neutral_face()

    # Shuffle and select two random expressions
    random.shuffle(expressions)
    selected_expressions = expressions[:2]

    # First expression prompt
    print(f"First expression: {selected_expressions[0]}")
    countdown(message="First expression in")
    if detect_expression_with_timer(selected_expressions[0]):
        print("First expression detected successfully!")
    else:
        print("Failed to detect the first expression.")
        return

    # Countdown before the second expression
    countdown(message="Next expression in")

    # Second expression prompt
    print(f"Second expression: {selected_expressions[1]}")
    if detect_expression_with_timer(selected_expressions[1]):
        print("Second expression detected successfully!")
    else:
        print("Failed to detect the second expression.")
        return

    print("Both expressions detected successfully!")

# Run the test
test_expression_prompts()

# Release resources
video.release()
cv2.destroyAllWindows()'
'''
'''
import cv2
import dlib
import numpy as np
import time
import random
import simpleaudio as sa
from scipy.spatial import distance

# Initialize webcam
video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye and mouth landmark indexes
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
MOUTH_LANDMARKS = list(range(48, 68))

# Updated expressions
expressions = ["Raised Eyebrows", "Shocked", "Happy", "Pout", "Frown", "Wink"]

# Global variables for dynamic thresholds
neutral_eyebrow_height = 0
neutral_mouth_height = 0
neutral_mouth_width = 0
neutral_ear = 0  # Eye Aspect Ratio for wink detection

# Sound files
ding_sound = "ding-36029.wav"  
buzzer_sound = "wrong-38598.wav"  

def play_sound(sound_file):
    try:
        wave_obj = sa.WaveObject.from_wave_file(sound_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound: {e}")

# Countdown function
def countdown(message="Starting in"):
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

    # Expression detection with dynamic thresholds
    if avg_eyebrow_height > neutral_eyebrow_height + 3:  # Raised Eyebrows
        return "Raised Eyebrows"
    elif avg_eyebrow_height < neutral_eyebrow_height - 3:  # Frown
        return "Frown"
    elif mouth_height > neutral_mouth_height + 5:  # Shocked
        return "Shocked"
    elif mouth_width > neutral_mouth_width + 10 and mouth_height < neutral_mouth_height - 2:  # Happy
        return "Happy"
    elif mouth_height < neutral_mouth_height - 2 and mouth_width < neutral_mouth_width - 5:  # Pout
        return "Pout"
    elif left_ear < neutral_ear - 0.2 or right_ear < neutral_ear - 0.2:  # Wink
        return "Wink"
    else:
        return "Neutral"

# Function to draw facial landmarks
def draw_landmarks(frame, landmarks):
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

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
                play_sound(ding_sound)
                time.sleep(0.5)  # Prevent double counting
            
        cv2.putText(frame, f"Blinks: {blink_count}/{blinks_required}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("BioPass Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    return True

# Expression detection with timer
def detect_expression_with_timer(expression):
    start_time = time.time()
    time_limit = 10  # 10 seconds per expression
    
    while (time.time() - start_time) < time_limit:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            landmarks = predictor(gray, face)
            detected_expression = detect_expression(landmarks)
            draw_landmarks(frame, landmarks)

            if detected_expression == expression:
                play_sound(ding_sound)
                return True

            cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Time left: {int(time_limit - (time.time() - start_time))}s", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("BioPass Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    play_sound(buzzer_sound)
    return False

# Main authentication flow
def start_authentication():
    # Initial countdown
    countdown()
    
    # Calibrate neutral face
    if not calibrate_neutral_face():
        play_sound(buzzer_sound)
        return

    # Blink detection
    blinks_required = random.randint(1, 3)
    print(f"Please blink {blinks_required} times!")
    
    if not blink_detection(blinks_required):
        print("Blink detection failed!")
        play_sound(buzzer_sound)
        return

    # Expression detection
    random.shuffle(expressions)
    selected_expressions = expressions[:2]  # Select 2 random expressions

    for i, expression in enumerate(selected_expressions):
        # Add countdown between expressions (except before first)
        if i > 0:
            countdown(message="Next expression in")
        
        print(f"Perform: {expression}")
        if not detect_expression_with_timer(expression):
            print(f"Failed to detect: {expression}")
            play_sound(buzzer_sound)
            return

    # Success
    print("Authentication successful!")
    play_sound(ding_sound)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Authentication Successful!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imshow("BioPass Authentication", frame)
    cv2.waitKey(3000)

# Run the authentication
start_authentication()

# Cleanup
video.release()
cv2.destroyAllWindows()
'''
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

'''
# Global variables for dynamic thresholds
neutral_eyebrow_height = 0
neutral_mouth_height = 0
neutral_mouth_width = 0
neutral_ear = 0  # Eye Aspect Ratio for wink detection
'''
# trying to find "universal" threshholds for expressions has been one of the most difficult parts of this project and has been extremely finnicky, so I have decided to use the neutral face calibration method to find the thresholds for each expression. This will allow for a more accurate and personalized detection of expressions based on the user's unique facial features.
EXPRESSION_THRESHOLDS = {
    "eyebrow_raise": 3.0,    # pixels
    "frown": -3.0,           # pixels
    "shocked": 5.0,          # pixels
    "happy_width": 10.0,     # pixels
    "happy_height": -2.0,    # pixels
    "pout_height": -2.0,     # pixels
    "pout_width": -5.0,      # pixels
    "wink_ear": 0.2          # ratio
}

# Sound files
'''
ding_sound = "ding_sound.wav"  
buzzer_sound = "buzzer_sound.wav"  
'''
# house all audio effects in a 'sounds' folder in my project directory
SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
ding_sound = os.path.join(SOUNDS_DIR, "ding_sound.wav")
buzzer_sound = os.path.join(SOUNDS_DIR, "buzzer_sound.wav")

'''
def play_sound(sound_file):
    wave_obj = sa.WaveObject.from_wave_file(sound_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()
'''
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

    # Expression detection with dynamic thresholds
    if avg_eyebrow_height > neutral_eyebrow_height + 3:  # Raised Eyebrows
        return "Raised Eyebrows"
    elif avg_eyebrow_height < neutral_eyebrow_height - 3:  # Frown
        return "Frown"
    elif mouth_height > neutral_mouth_height + 5:  # Shocked
        return "Shocked"
    elif mouth_width > neutral_mouth_width + 10 and mouth_height < neutral_mouth_height - 2:  # Happy
        return "Happy"
    elif mouth_height < neutral_mouth_height - 2 and mouth_width < neutral_mouth_width - 5:  # Pout
        return "Pout"
    elif left_ear < neutral_ear - 0.2 or right_ear < neutral_ear - 0.2:  # Wink
        return "Wink"
    else:
        return "Neutral"

# Function to draw facial landmarks
def draw_landmarks(frame, landmarks):
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

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

# Expression detection with timer
def detect_expression_with_timer(expression):
    start_time = time.time()
    time_limit = 15  # 15 seconds per expression
    
    while (time.time() - start_time) < time_limit:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            landmarks = predictor(gray, face)
            detected_expression = detect_expression(landmarks)
            draw_landmarks(frame, landmarks)

            if detected_expression == expression:
                play_sound(ding_sound)
                return True

            cv2.putText(frame, f"Make: {expression}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Time left: {int(time_limit - (time.time() - start_time))}s", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("BioPass Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    play_sound(buzzer_sound)
    return False

# Main authentication flow
def start_authentication():
    # Initial countdown
    countdown()
    
    # Calibrate neutral face
    if not calibrate_neutral_face():
        return

    # Blink detection
    blinks_required = random.randint(1, 3)
    print(f"Please blink {blinks_required} times!")
    
    if not blink_detection(blinks_required):
        print("Blink detection failed!")
        play_sound(buzzer_sound)
        return

    # Expression detection
    random.shuffle(expressions)
    selected_expressions = expressions[:2]  # Select 2 random expressions

    for i, expression in enumerate(selected_expressions):
        # Add countdown between expressions (except before first)
        if i > 0:
            countdown(message="Next expression in")
        
        print(f"Perform: {expression}")
        if not detect_expression_with_timer(expression):
            print(f"Failed to detect: {expression}")
            return

    # Success
    print("Authentication successful!")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Authentication Successful!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imshow("BioPass Authentication", frame)
    cv2.waitKey(3000)

# Run the authentication
start_authentication()

# Cleanup
video.release()
cv2.destroyAllWindows()