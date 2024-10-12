import cv2
import face_recognition
import numpy as np

# Load a sample picture and learn how to recognize it.
# Replace 'your_image.jpg' with the actual path of a known image of the subject.
known_image = face_recognition.load_image_file("your_image.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Create an array of known face encodings and their names
known_face_encodings = [known_face_encoding]
known_face_names = ["Your Name"]  # Replace with the actual name of the subject

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

# Access webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (used by OpenCV) to RGB color (used by face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up the face locations since the frame we detected was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
