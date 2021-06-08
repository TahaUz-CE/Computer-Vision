import cv2
import numpy as np
import matplotlib.pyplot as plt

# Img Load
img = cv2.imread('Kopek.PNG')
plt.imshow(img)
plt.show()

# Color RGB
img_Rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_Rgb)
plt.show()
new_img = img_Rgb.copy()

# Img Upside down
new_img = cv2.flip(new_img,0)
plt.imshow(new_img)
plt.show()

# Drawing rectangle on IMG
cv2.rectangle(img_Rgb,pt1=(150,75),pt2=(350,275),color=(255,0,0),thickness=10)
plt.imshow(img_Rgb)
plt.show()


# 250,75 150,275 350,275
# Drawing a shape with 3 corners include dog head .
vertices = np.array([[250,75],[150,275],[350,275]],np.int32)
#print(vertices)
#print(vertices.shape)
pts = vertices.reshape((-1,1,2))
#print(pts.shape)

# Load Clear IMG
img_Rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Drawing a shape with 3 corners.
cv2.polylines(img_Rgb,[pts],isClosed=True,color=(0,0,255),thickness=20)
plt.imshow(img_Rgb)
plt.show()

cv2.fillPoly(img_Rgb,[pts],color=(0,0,255))
plt.imshow(img_Rgb)
plt.show()

# Drawing Circle with mouse click on IMG.
def create_circle(event,x,y,flags,param):

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,0,255),thickness=10)

img = cv2.imread('Kopek.PNG')
cv2.namedWindow(winname='dog')
cv2.setMouseCallback('dog',create_circle)

# Last Img Close(ESC button)
while True:

    cv2.imshow('dog',img)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

