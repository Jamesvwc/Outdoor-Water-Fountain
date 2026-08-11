import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

if (webcam.isOpened()):
    print("Video has been opened")
    fps = webcam.get(cv2.CAP_PROP_FPS)
    frame_count = webcam.get(cv2.CAP_PROP_FRAME_COUNT)
    width = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    videoType = webcam.get(cv2.CAP_PROP_FRAME_TYPE)
    print("FPS is:", fps)
    print("Frame count is:", frame_count)
    print("Width of the video is:", width)
    print("Height of the video is:", height)
    print("The type of video (whatever that means) is", videoType)
else:
    print("Video has not been opened")

webcam.release()