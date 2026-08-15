import cv2 as cv
import numpy as np
import time
from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                             InferVStreams, ConfigureParams,
                             InputVStreamParams, OutputVStreamParams,
                             FormatType)

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]
ANIMAL_CLASSES = {'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'}

HEF_PATH = "resources/models/hailo8/yolov8m.hef"  # 26 TOPS variant
DETECTION_THRESHOLD = 0.5

# ---- Set up the NPU + model once, outside the loop ----
hef = HEF(HEF_PATH)
target = VDevice()

configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = target.configure(hef, configure_params)[0]
network_group_params = network_group.create_params()

input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
model_input_h, model_input_w, _ = input_vstream_info.shape  # typically 640x640

input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
infer_pipeline = InferVStreams(network_group, input_vstreams_params, output_vstreams_params)


def run_yolo(frame_bgr):
    """Runs a frame through the Hailo NPU. Returns list of
    (label, confidence, (x1, y1, x2, y2)) in original frame coordinates."""
    h, w = frame_bgr.shape[:2]
    resized = cv.resize(frame_bgr, (model_input_w, model_input_h))
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    input_data = {input_vstream_info.name: np.expand_dims(rgb, axis=0)}

    with network_group.activate(network_group_params):
        results = infer_pipeline.infer(input_data)

    # NMS is baked into the HEF: output is per-class [y_min, x_min, y_max, x_max, score], normalized 0-1
    raw = results[output_vstream_info.name][0]

    detections = []
    for class_id, class_dets in enumerate(raw):
        if class_id >= len(COCO_CLASSES):
            continue
        label = COCO_CLASSES[class_id]
        for det in class_dets:
            y_min, x_min, y_max, x_max, score = det
            if score < DETECTION_THRESHOLD:
                continue
            x1, y1, x2, y2 = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
            detections.append((label, float(score), (x1, y1, x2, y2)))
    return detections


# ---- Main loop: no background subtraction needed ----
camera = cv.VideoCapture(0)
time.sleep(1)

while True:
    success, frame = camera.read()
    if not success:
        continue

    frame = cv.resize(frame, (1280, 720))
    detections = run_yolo(frame)

    for label, score, (x1, y1, x2, y2) in detections:
        color = (0, 255, 0) if label in ANIMAL_CLASSES else (0, 165, 255)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"{label} {score:.0%}", (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv.imshow("Animal Detection", frame)
    if cv.waitKey(1) & 0xFF != 255:
        break

camera.release()
cv.destroyAllWindows()