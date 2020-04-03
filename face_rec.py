import os
import cv2
import face_recognition

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
FRAME_THICKNESS = 3
FONT_THICKNESS = 1
TOLERANCE = 0.5 # low tolerance = 100% sure match. 0.6 is default in face-rec.
MODEL = "cnn" # convolution neural network. can also use "hog", Histogram of Oriented Gradients.

print("loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}") # load image
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("working on unknown faces...")
# this will take the unknown faces and compare them with known ones.
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL) # location will check faces coordinates
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 0, 0] # black
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            top_left = (face_location[3], face_location[2]) # smaller rectangle for font
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (200,200,200), FONT_THICKNESS)
        else:
            print(f"Match not found")

    cv2.imshow(filename, image)
    cv2.waitKey(1000) # 1sec
    #cv2.destroyWindow(filename)
