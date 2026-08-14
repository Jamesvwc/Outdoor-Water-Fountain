import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time

#load and preprocess the image
def load_and_preprocess_frame(frame):
    captured_frame = cv.imread(frame) #static image from camera upon motion detection
    resized_frame = cv.resize(captured_frame, (1280, 720)) #resize images from camera for model to read
    gray_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY) #convert from bgr to rgb to ensure expected format is correct
    return resized_frame, gray_frame

#subtract the static image and the new image
def subtract_images(img1, img2):
    diff = cv.absdiff(img1, img2)
    _, thresh = cv.threshold(diff, 85, 255, cv.THRESH_BINARY)
    return diff, thresh

img_path1 = 'testphoto1.jpg'
img_path2 = 'testphoto2.jpg'

image1, gray_image1 = load_and_preprocess_frame(img_path1)
image2, gray_image2 = load_and_preprocess_frame(img_path2)

#subtract the images
diff, thresh = subtract_images(gray_image1, gray_image2)

#plot the images
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.title('Static image')
plt.imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Test image')
plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Difference')
plt.imshow(diff, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Threshold Difference')
plt.imshow(thresh, cmap='gray')
plt.axis('off')

# plt.show()

dilated_image = cv.dilate(thresh, None, iterations=2)
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')
# plt.show()

contours = cv.findContours(dilated_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours) #normalize the contours

#iterate through the contours
for c in contours:
    if (cv.contourArea(c) < 700):
        continue
    #else, get the bounding box coordinates
    (x, y, w, h) = cv.boundingRect(c)
    cv.rectangle(image2, (x,y), (x+w, y+h), (0, 255, 0), 2)

# cv.imshow('test', image2)
# cv.waitKey(5000)
# cv.destroyAllWindows()

camera = cv.VideoCapture(0)
fps = camera.get(cv.CAP_PROP_FPS)
static_frame = None

while True:
    success, frame = camera.read()
    if (success == True):
        frame = cv.resize(frame, (1280, 720))
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if (static_frame is None):
            static_frame = gray_frame #first frame of webcam is the static frame
            continue

        diff, thresh = subtract_images(static_frame, gray_frame)
        dilated_image = cv.dilate(thresh, None, iterations=2)

        contours = cv.findContours(dilated_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours) #normalize the contours

        #iterate through the contours
        for c in contours:
            if (cv.contourArea(c) < 1500):
                continue
            #else, get the bounding box coordinates
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow("Motion Detection", frame)

        if (cv.waitKey(1) & 0xFF != 255):
            break
        time.sleep(0.05)
