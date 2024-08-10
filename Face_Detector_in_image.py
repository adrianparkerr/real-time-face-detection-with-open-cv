import cv2
from random import randrange

# 1. Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
# use cv2(opencv).CasacadeClassifier : a fct that makes a classifier ( a detector)
trained_face_data = cv2.CascadeClassifier(
    "./haarcascade_frontalface_default.xml")


trained_eyes_data = cv2.CascadeClassifier(
    "./haarcascade_eye.xml")

# 2. Choose an image ,to detect a face : use cv2.imread fct
img = cv2.imread("jake.jpg")


# 3. Must convert to grayscale
greyImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# 4. Detect Face
# the detection is done using the cv::CascadeClassifier::detectMultiScale method, which returns boundary rectangles for the detected faces (or eyes).
# MultiScale => will detect a face no matter how small or big (no matter the scale...)

# ==> returns the face coordinates ,which will be used to draw to rectangle surrounding it
face_coordinates = trained_face_data.detectMultiScale(
    greyImage, minSize=[80, 90])

print(face_coordinates)


#
# 4. Detect Eyes


eye_coordinates = trained_eyes_data.detectMultiScale(
    greyImage, )

print("eye coordinates : \n", eye_coordinates)

# (x, y, w, h) = face_coordinates[1]

# 5. Draw rectangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(255),
                  randrange(255), randrange(255)), 5)


#
# 5. Draw rectangles around the eyes
for (x, y, w, h) in eye_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,
                  randrange(255), 0), 2)


#imgshow("name of the window",image)


cv2.imshow("Face Detector with Python", img)

cv2.waitKey()  

# makes program wait until a key is touched


print("code completed")
