import cv2
import matplotlib.pyplot as plt
import numpy as np

# Img Loader function
def display(title,img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    ax.set_title('Zurafa '+title)
    plt.show()

# Img Convert Color RGB

img = cv2.imread('zurafa.jpg')
img_color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
display(title='IMG Convert Color RGB',img=img_color)

# ThreshHolding

img = cv2.imread('zurafa.jpg',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
display(img=thresh1,cmap='gray',title="ThreshHolding")

# Filter2D

img = cv2.imread('zurafa.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
display(img=img,title="Filter2D HSV")

kernel = np.ones(shape=(4,4),dtype=np.float32)/10

img = cv2.imread('zurafa.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

result = cv2.filter2D(img,-1,kernel)
display(img=result,title="Filter2D Blur")

#Sobel

img = cv2.imread('zurafa.jpg',0)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

display(img=sobelx,cmap='gray',title="Sobel X")

# Histograms

img = cv2.imread('zurafa.jpg')

color = ['b','g','r']

for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)

plt.title('Zurafa Histogram Red, Blue and Green Channel')
plt.show()














