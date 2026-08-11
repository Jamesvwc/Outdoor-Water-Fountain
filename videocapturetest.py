import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

if (webcam.isOpened()):
    print("Video has been opened")
    fps = webcam.get(cv2.CAP_PROP_FPS)
    frame_count = webcam.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS is:", fps)
    print("Frame count is:", frame_count)
    print("Width of the video is:", width)
    print("Height of the video is:", height)
    frame_size = (width, height)

    webcam_out = cv2.VideoWriter('webcam.avi', 
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        30,
        frame_size)

else:
    print("Video has not been opened")

while (webcam.isOpened()):
    ret, frame = webcam.read()
    if ret == True:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_stack = np.stack([frame] * 3, axis = -1)
        webcam_out.write(frame_stack)

        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(5)
        if (key == ord('q')):
            break
    else:
        break

webcam.release()
webcam_out.release()