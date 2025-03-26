import cv2
import os
import uuid
import numpy as np
import face_recognition
import speech_recognition as sr
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Directory to store captured face images
CAPTURED_FACE_DIR = "captured_faces"
if not os.path.exists(CAPTURED_FACE_DIR):
    os.makedirs(CAPTURED_FACE_DIR)

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Keep track of recognized faces to prevent multiple announcements
seen_faces = set()

# Load previously saved face images
for filename in os.listdir(CAPTURED_FACE_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(CAPTURED_FACE_DIR, filename)
        image = face_recognition.load_image_file(filepath)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # Ensure the encoding is valid
            known_face_encodings.append(encoding[0])
            name = filename.split("_")[0]  # Extract the name from filename
            known_face_names.append(name)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Open webcam (0 = Default Camera)
cap = cv2.VideoCapture(0)

captured = False  # Flag to ensure we capture only one face
exit_announced = False  # Ensures "Finish Shopping" is spoken only once

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # Convert frame to RGB (required for face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face encodings in the current frame
    face_encodings = face_recognition.face_encodings(rgb_frame)

    for (x, y, w, h), face_encoding in zip(faces, face_encodings):
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

            # Announce the name only once per session
            if name not in seen_faces:
                print(f"Welcome back, {name}!")
                engine.say(f"Welcome back, {name}!")
                engine.runAndWait()
                seen_faces.add(name)  # Mark as seen

        else:
            if not captured:  # Capture only if it's a new face
                # Crop the detected face
                face_crop = frame[y:y + h, x:x + w]
                cv2.imshow("Captured Face", face_crop)
                cv2.waitKey(1)  # Briefly display the captured face

                # Ask the user to say their name
                print("Please say your name... (or say 'finish shopping' to exit)")
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce noise
                    try:
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Increased timeout
                        name = recognizer.recognize_google(audio).strip().lower()
                        
                        # Check if the user said "finish shopping"
                        if name == "finish shopping" and not exit_announced:
                            print("Exiting... Thank you for shopping!")
                            engine.say("Exiting... Thank you for shopping!")
                            engine.runAndWait()
                            exit_announced = True
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()

                        print(f"Recognized Name: {name}")
                    except sr.UnknownValueError:
                        print("Sorry, could not understand the name. Using 'Unknown'.")
                        name = "Unknown"
                    except sr.RequestError:
                        print("Could not connect to speech recognition service.")
                        name = "Unknown"
                    except sr.WaitTimeoutError:
                        print("No speech detected. Using 'Unknown'.")
                        name = "Unknown"

                # Save the new face
                safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).rstrip()
                filename = os.path.join(CAPTURED_FACE_DIR, f"{safe_name}_{uuid.uuid4().hex}.jpg")
                cv2.imwrite(filename, face_crop)
                print(f"Face captured and saved as: {filename}")

                # Add the new face to known faces
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

                # Announce the newly captured name
                engine.say(f"Nice to meet you, {name}!")
                engine.runAndWait()

                captured = True  # Ensure only one face is captured
                break  # Exit the loop after capturing

        # Draw a rectangle around the face with the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Face Recognition", frame)

    # Check if the user wants to exit by saying "finish shopping" (Only once)
    if not exit_announced:
        print("Say 'finish shopping' to exit.")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)
                command = recognizer.recognize_google(audio).strip().lower()
                if command == "finish shopping":
                    print("Exiting... Thank you for shopping!")
                    engine.say("Exiting... Thank you for shopping!")
                    engine.runAndWait()
                    exit_announced = True
                    break
            except sr.UnknownValueError:
                pass
            except sr.WaitTimeoutError:
                pass

    # Break if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
