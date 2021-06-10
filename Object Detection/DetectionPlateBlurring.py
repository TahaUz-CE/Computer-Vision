# Import Library
import cv2
import matplotlib.pyplot as plt

# Load IMG
img = cv2.imread('RussianPlate.jpg')

def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    plt.show()

#display(img)

# Load Cascade Classifier
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# This Function Detect Plate and Drawing Rectangle Include Plate on IMG
def detect_plate(img):
    plate_img = img.copy()
    # Returns the coordinates of the plate
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)
    # Drawing Rectangle Include Plate on IMG
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255),4)

    return plate_img


result = detect_plate(img)
display(result)

# This Function Detect Plate and Blurring Include Plate on IMG
def detect_and_blur_plate(img):
    plate_img = img.copy()
    roi = img.copy()
    # Returns the coordinates of the plate
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)
    # Blurring Include Plate on IMG
    for (x,y,w,h) in plate_rects:

        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,25)

        plate_img[y:y+h,x:x+w] = blurred_roi

    return plate_img


result = detect_and_blur_plate(img)
display(result)