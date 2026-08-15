import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
import tensorflow as tf
import time

# class AnimalDetection:

#     def __init__(self, camera_index=0, min_area=2000):
#         self.captured = cv.VideoCapture(camera_index)
#         cam_width = int(self.captured.get(cv.CAP_PROP_FRAME_WIDTH))
#         cam_height = int(self.captured.get(cv.CAP_PROP_FRAME_HEIGHT))
#         self.resize_dims = (cam_width, cam_height)
#         self.min_area = min_area
#         self.fgbg = cv.createBackgroundSubtractorMOG2(history=750, varThreshold=16, detectShadows=True)

# def read_frame(self):
#     ret, frame = self.captured.read()
#     if (ret == False): #camera is capturing feed
#         return None
#     return frame

# def initialize_camera(self):
#     if (read_frame == None):
#         #sleep for 1 second then retry function
#         read_frame(self)
#     return

# def main_camera(self):
#     while (self.captured.isOpened()):
#         ret, frame = self.captured.read()
#         if (ret == True):
#             frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #gray when no motion is detected to ease processing
#             if (motion_detected):
#                 load_and_preprocess_frame(frame)
#         break
#     return

# def motion_detected(self, frame):
#     frame = self.captured.read()
#     return

# def load_and_preprocess_frame(frame):
#     captured_frame = cv.imread(frame) #static image from camera upon motion detection
#     resized_frame = cv.resize(captured_frame, (1280, 720)) #resize images from camera for model to read
#     gray_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY) #convert from bgr to rgb to ensure expected format is correct
#     orig_frame = resized_frame
#     return orig_frame, gray_frame

# normalized_frame = rgb_frame.astype(np.float32) / 255.0 #normalize pixel values
# np.expand_dims(normalized_frame, axis=0) #returns normalized image with fourth dimension indicating batch size

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

# #plot the images
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

# # plt.show()

# dilated_image = cv.dilate(thresh, None, iterations=2)
# plt.imshow(dilated_image, cmap='gray')
# plt.axis('off')
# # plt.show()

# contours = cv.findContours(dilated_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)

# #iterate through the contours
# for c in contours:
#     if (cv.contourArea(c) < 700):
#         continue
#     #else, get the bounding box coordinates
#     (x, y, w, h) = cv.boundingRect(c)
#     cv.rectangle(image2, (x,y), (x+w, y+h), (0, 255, 0, 2))

# cv.imshow('test', image2)
# cv.waitKey(10000)
# cv.destroyAllWindows()

# camera = cv.VideoCapture(0)
# static_frame = None

class AnimalDetection:
    
    def __init__(self, camera_index=0, model_dims=(300, 300), min_area=2000):
        self.captured = cv.VideoCapture(camera_index)
        self.resize_dims = model_dims
        self.min_area = min_area
        self.fgbg = cv.createBackgroundSubtractorMOG2(history=750, varThreshold=40, detectShadows=True)

    def get_frame(self):
        success, frame = self.captured.read()
        if (success == True):
            return frame
        return None
    
    def motion_detected(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #convert from bgr to gray for easier processing
        masked_image = self.fgbg.apply(gray_frame, learningRate=0.001)
        _, polished_image = cv.threshold(masked_image, 200, 255, cv.THRESH_BINARY)
        polished_image = cv.morphologyEx(polished_image, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        polished_image = cv.dilate(polished_image, np.ones((5,5), np.uint8), iterations=2)

        # #contours
        contours = cv.findContours(polished_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        motion = False
        results = []
        for c in contours:
            if (cv.contourArea(c) < self.min_area):
                continue #ie. not enough motion
            motion = True
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            results.append((x, y, w, h))

        cv.imshow('test', frame)
        cv.waitKey(50)
        self.last_mask = polished_image
        self.last_results = results
        return motion
    
    # def motion_detected(self, frame):
    #     fg_mask = self.fgbg.apply(frame)
    #     fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    #     contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #     for c in contours:
    #         if cv.contourArea(c) > self.min_area:
    #             return True
    #     cv.imshow('video', frame)
    #     cv.waitKey(40)
    #     return False
    
    def show_camera_feed(self, frame):
        display_frame = frame.copy()
        for (x, y, w, h) in getattr(self, 'last_boxes', []):
            cv.rectangle(display_frame, (x,y), (x+w, y+h), (0, 255, 0, 2))
    
    def preprocess_frame_for_model(self, frame):
        resized_frame = cv.resize(frame, self.resize_dims) #resize images from camera for model to read
        rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_GRAY2RGB)
        normalized_frame = rgb_frame.astype(np.float32) / 255.0
        return np.expand_dims(normalized_frame, axis=0)

    def release(self):
        self.captured.release()

# class AnimalClassification:

#     def __init__(self, model_path, label_path, threshold=0.5):
#         self.interpreter = tf.lite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output.details()
#         self.labels = self._load_labels(label_path)
#         self.threshold = threshold

#     def _load_labels(self, path):
#         with open(path, 'r') as f:
#             return [line_strip() for line in f.readlines()]
        
#     def predict(self, preprocessed_frame):
#         self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_frame)
#         self.interpreter.invoke()
#         output = self.interpreter.get_tensor(self.output_details[0]['index'])
#         results = []
#         for i, score in enumerate(output[0]):
#             if score > self.threshold:
#                 results.append((self.labels[i], float(score)))
#         return sorted(results, key=lambda x: x[1], reverse=True)

def main():
    camera = AnimalDetection(camera_index=0)

    while True:
        frame = camera.get_frame()
        if (frame is None): #camera has not been opened yet
            continue

        if (camera.motion_detected(frame)):
            continue
        
if __name__ == "__main__":
    main()
