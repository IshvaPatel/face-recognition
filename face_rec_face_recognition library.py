import cv2
import face_recognition
import os
import glob
import numpy as np

dir_path="C:/face_rec/people"
images=glob.glob(os.path.join(dir_path,"*.*"))
face_locations1=[]
face_encodings1=[]
names=[]

for i in images:
    image = cv2.imread(i)
    basename = os.path.basename(i)
    known_face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    f_locations = face_recognition.face_locations(known_face)
    f_encodings = face_recognition.face_encodings(known_face, f_locations)
    face_locations1.extend(f_locations)
    face_encodings1.extend(f_encodings)
    names.append(basename)

video = cv2.VideoCapture("C:/face_rec/title-artist.mp4")


# Face encodings of the known face


while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("End of video.")
        break

    # Resize and convert the frame to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for i, face_encoding in enumerate(face_encodings):
        
        matches = face_recognition.compare_faces(face_encodings1, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(face_encodings1, face_encoding)
        best_match_index = np.argmin(face_distances)
        name='Unknown'
        top, right, bottom, left = face_locations[i]
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 0, 200), 4)
        #name=names[best_match_index]
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left * 4, bottom * 4 + 35), font, 0.8, (255, 255, 255), 1)

    # Rest of the matching and drawing code...


        if matches[best_match_index] and 0 <= best_match_index < len(face_locations):
            top, right, bottom, left = face_locations[best_match_index]
            cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 0, 200), 4)
            name=names[best_match_index]
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left * 4, bottom * 4 + 35), font, 0.8, (255, 255, 255), 1)


    # Display the current frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
