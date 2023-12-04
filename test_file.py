import cv2
import face_recognition

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error in video")
        break

    frame = cv2.flip(frame, 1)

    # Identify face locations
    face_locations = face_recognition.face_locations(frame)

    # Identify face landmarks
    face_landmarks_list = face_recognition.face_landmarks(frame)

    for face_landmarks in face_landmarks_list:
        # Loop over each facial feature (eye, nose, mouth, lips, etc)
        for name, list_of_points in face_landmarks.items():
            # Draw a line on the frame joining each point
            for point in list_of_points:
                cv2.circle(frame, point, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
