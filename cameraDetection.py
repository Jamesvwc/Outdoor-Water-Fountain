import cv2 as cv
import numpy as np

def __init__(self, camera_index=0, min_area=2000):
    self.captured = cv.VideoCapture(camera_index)
    cam_width = int(self.captured.get(cv.CAP_PROP_FRAME_WIDTH))
    cam_height = int(self.captured.get(cv.CAP_PROP_FRAME_HEIGHT))
    self.resize_dims = (cam_width, cam_height)
    self.min_area = min_area
    self.fgbg = cv.createBackgroundSubtractorMOG2(history=750, varThreshold=16, detectShadows=True)

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

def motion_detected(self):
    frame = self.captured.read()
    return

def load_and_preprocess_frame(self, frame):
    frame = cv.imread(frame)
    frame = cv.resize(frame, self.resize_dims) #resize images from camera for model to read
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #convert from bgr to rgb to ensure expected format is correct
    normalized_frame = rgb_frame.astype(np.float32) / 255.0 #normalize pixel values
    return np.expand_dims(normalized_frame, axis=0) #returns normalized image with fourth dimension indicating batch size

def subtract_background(self, frame):
    
    return