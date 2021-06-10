# Import Library
import cv2

# Load Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):

    face_img = img.copy()
    # Detect face with cascade in detectMultiScale function
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    # Drawing Rectangle and Write on the frame
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)

    return face_img


# Load Cascade Classifier
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_eye(img):

    face_img = img.copy()
    # Detect face with cascade in detectMultiScale function
    eye_rects = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=6)
    # Drawing Rectangle and Write on the frame
    for (x,y,w,h) in eye_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,0,255),10)

    return face_img

# Video Capture Setup
cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read(0)

    frame = detect_face(frame)

    frame = detect_eye(frame)

    cv2.imshow('Video Face Detect',frame)

    # Close Window => ESC button
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()