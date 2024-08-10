import cv2
from random import randrange

# 1. Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
# use cv2(opencv).CasacadeClassifier : a fct that makes a classifier ( a detector)
trained_face_data = cv2.CascadeClassifier(
    "./haarcascade_frontalface_default.xml")


# 2. to capture video from web cam
# img = cv2.imread("jake.jpg")
# 0 is for default webcam
webcam = cv2.VideoCapture(0)


# Iterate frames infinitely
while True:

    # Read current frame
    # .read() returns two values : a boolean the frame being reda
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    greyImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Detect Face
    face_coordinates = trained_face_data.detectMultiScale(greyImage)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255),
                                                  randrange(255), randrange(255)), 2)
    cv2.imshow("Face Detector with Python", frame)
    key = cv2.waitKey(1)  # makes program wait until a key is touched

    # quit if q is pressed
    if key == 81 or key == 113:
        break

# release webcam
webcam.release()

""""
print(face_coordinates)
# (x, y, w, h) = face_coordinates[1]
# 5. Draw rectangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(255),
                  randrange(255), randrange(255)), 2)



#imgshow("name of the window",image)
cv2.imshow("Face Detector with Python", img)
cv2.waitKey()  # makes program wait until a key is touched


print("code completed")
"""
