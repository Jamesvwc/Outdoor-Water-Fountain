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

# #subtract the static image and the new image
# def subtract_images(img1, img2):
#     diff = cv.absdiff(img1, img2)
#     _, thresh = cv.threshold(diff, 85, 255, cv.THRESH_BINARY)
#     return diff, thresh

# img_path1 = 'testphoto1.jpg'
# img_path2 = 'testphoto2.jpg'

# image1, gray_image1 = load_and_preprocess_frame(img_path1)
# image2, gray_image2 = load_and_preprocess_frame(img_path2)

# #subtract the images
# diff, thresh = subtract_images(gray_image1, gray_image2)

#plot the images
# plt.figure(figsize=(15,10))
# plt.subplot(2, 2, 1)
# plt.title('Static image')
# plt.imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.title('Test image')
# plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.title('Difference')
# plt.imshow(diff, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.title('Threshold Difference')
# plt.imshow(thresh, cmap='gray')
# plt.axis('off')

# plt.show()

# dilated_image = cv.dilate(thresh, None, iterations=2)
# plt.imshow(dilated_image, cmap='gray')
# plt.axis('off')
# plt.show()

# contours = cv.findContours(dilated_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours) #normalize the contours

# #iterate through the contours
# for c in contours:
#     if (cv.contourArea(c) < 700):
#         continue
#     #else, get the bounding box coordinates
#     (x, y, w, h) = cv.boundingRect(c)
#     cv.rectangle(image2, (x,y), (x+w, y+h), (0, 255, 0), 2)

# cv.imshow('test', image2)
# cv.waitKey(5000)
# cv.destroyAllWindows()

# def merge_boxes(boxes, distance_thresh=20):
#     if not boxes:
#         return []
#     boxes = sorted(boxes, key=lambda b: b[0])
#     merged = [list(boxes[0])]
#     for (x, y, w, h) in boxes[1:]:
#         mx, my, mw, mh = merged[-1]
#         # check if this box is close to / overlaps the last merged box
#         if x <= mx + mw + distance_thresh and y <= my + mh + distance_thresh and y + h >= my - distance_thresh:
#             nx = min(mx, x)
#             ny = min(my, y)
#             nx2 = max(mx + mw, x + w)
#             ny2 = max(my + mh, y + h)
#             merged[-1] = [nx, ny, nx2 - nx, ny2 - ny]
#         else:
#             merged.append([x, y, w, h])
#     return merged

# def merge_contours_by_mask(contours, frame_shape, dilate_kernel_size=35, dilate_iterations=3):
#     # 1. blank mask, same size as the frame (single channel)
#     mask = np.zeros(frame_shape[:2], dtype=np.uint8)

#     # 2. draw every contour onto the mask, filled in white
#     cv.drawContours(mask, contours, -1, 255, thickness=cv.FILLED)

#     # 3. dilate to bridge small gaps between nearby/separate contours
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
#     mask = cv.dilate(mask, kernel, iterations=dilate_iterations)
#     mask = cv.morphologyEx(dilated_image, cv.MORPH_CLOSE, kernel, iterations=2)

#             # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
#         # dilated_image = cv.dilate(thresh, kernel, iterations=3)
#         # dilated_image = cv.morphologyEx(dilated_image, cv.MORPH_CLOSE, kernel, iterations=2)

#     # 4. re-find contours on the merged mask — touching blobs are now one contour
#     merged_contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     merged_contours = imutils.grab_contours(merged_contours)

#     return merged_contours

#subtract the static image and the new image
def subtract_images(img1, img2):
    diff = cv.absdiff(img1, img2)
    _, thresh = cv.threshold(diff, 85, 255, cv.THRESH_BINARY)
    return diff, thresh

camera = cv.VideoCapture(0)
# fps = camera.get(cv.CAP_PROP_FPS)
time.sleep(1) #give the camera time to turn on
static_frame = None

while True:
    success, frame = camera.read()

    if (success == True):
        # frame, gray_frame = load_and_preprocess_frame(frame)
        frame = cv.resize(frame, (1280, 720))
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.GaussianBlur(gray_frame, (21, 21), 0)

        if (static_frame is None):
            static_frame = gray_frame #first frame of webcam is the static frame
            continue

        diff, thresh = subtract_images(static_frame, gray_frame)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
        dilated_image = cv.dilate(thresh, kernel, iterations=3)
        dilated_image = cv.morphologyEx(dilated_image, cv.MORPH_CLOSE, kernel, iterations=2)
        # dilated_image = cv.dilate(thresh, None, iterations=2)

        contours = cv.findContours(dilated_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours) #normalize the contours

        #iterate through the contours
        # boxes = []
        contours = [c for c in contours if cv.contourArea(c) >= 2000]

        # merged_contours = merge_contours_by_mask(contours, frame.shape)

        for c in contours:
            if (cv.contourArea(c) < 2000):
                continue
            #else, get the bounding box coordinates
            # boxes.append(cv.boundingRect(c))
        # for (x, y, w, h) in merge_boxes(boxes):
        #     cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow("Motion Detection", frame)

        if (cv.waitKey(1) & 0xFF != 255):
            break
        time.sleep(0.05)
 