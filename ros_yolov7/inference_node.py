import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

import cv2
import cv_bridge

# ---------------------------------------------------------------------------- #
#                                 SANDBOX BEGIN                                #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                YOLOV7 IMPORTS                                #
# ---------------------------------------------------------------------------- #
import argparse
import time
from pathlib import PurePath

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

opt_weights     = "/home/roar/ART/perception/model_trials/yolov7_iac/runs/train/yolov7_p5_transfer_flc8/weights/best.pt" #TODO
opt_img_size    = 1056
opt_conf_thres  = 0.25
opt_iou_thres   = 0.45
opt_device      = ''

class YOLOv7():
    def __init__(self, weights, img_size, classes, trace=True, confidence=0.25, iou_threshold=0.45):
        with torch.no_grad():
            self.weights        = opt_weights 
            self.imgsz          = opt_img_size
            self.trace          = trace
            self.confidence     = confidence
            self.iou_threshold  = iou_threshold
            self.classes        = None

            # Initialize
            set_logging()
            self.device     = select_device(opt_device)
            self.half       = self.device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            self.model      = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride     = int(self.model.stride.max())  # model stride
            self.imgsz      = check_img_size(self.imgsz, s=self.stride)  # check img_size

            if self.trace:
                self.model = TracedModel(self.model, self.device, self.imgsz)

            if self.half:
                self.model.half()  # to FP16

            cudnn.benchmark = True  # set True to speed up constant image size inference

            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            self.warmup_required = True

            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
            self.old_img_w = self.old_img_h = self.imgsz
            self.old_img_b = 1
        
        print("YOLO Init DONE!!!\n Waiting for Node Init to finish:\n")
    
    def img_preproc(self, img0):
        img = letterbox(img0,  self.imgsz, stride=self.stride)[0]
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0

    def run_inference(self, img, im0s, msg, publisher):
        with torch.no_grad():
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.warmup_required and self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=False)[0]
                
                self.warmup_required = False

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.confidence, self.iou_threshold, classes=self.classes, agnostic=False)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        msg_copy = msg
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        msg_copy.bbox.center.x          = xywh[0]
                        msg_copy.bbox.center.y          = xywh[1]
                        msg_copy.bbox.size_x            = xywh[2]
                        msg_copy.bbox.size_y            = xywh[3]
                        hypothesis                      = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.score     = float(conf.cpu().numpy())
                        hypothesis.hypothesis.class_id  = str(cls)
                        msg_copy.results.append(hypothesis)
                        publisher.publish(msg_copy)

                        plot_one_box(xyxy, im0s, label='yolov7', line_thickness=3)
            
            cv2.imshow('YOLO', im0s)
            cv2.waitKey(1)



class InferenceYOLOv7(Node):

    def __init__(self):
        super().__init__('inference_yolov7')

        # -------------------------------- QOS Profile ------------------------------- #
        self.qos_profile =  QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----------------------------- Camera Subscriber ---------------------------- #
        self.image_sub  = self.create_subscription(
            # msg_type    = CompressedImage,
            msg_type    = Image,
            # topic       = '/camera/front_left_center/image/compressed',
            topic       = '/vimba_front_left_center/image',
            callback    = self.image_callback,
            qos_profile = self.qos_profile
        )
        self.image_sub # prevent unused variable warning
        
        # ---------------------------- Detection Publisher --------------------------- #
        self.detection_publisher = self.create_publisher(
            msg_type=Detection2D,
            topic='/front_left_center/detections',
            qos_profile=self.qos_profile
        )

        # -------------------------------- CV2 Bridge -------------------------------- #
        self.bridge = cv_bridge.CvBridge()

        # -------------------------------- Load Model -------------------------------- #
        self.yolo = YOLOv7(weights=opt_weights, img_size=2080, classes='')

        print("Node Init Done!!! \n Waiting for Images:")

    def image_callback(self, msg):
        # self.cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img, img0   = self.yolo.img_preproc(self.cv_img)

        det_msg = Detection2D()
        det_msg.header = msg.header
        msg = self.yolo.run_inference(img, img0, det_msg, self.detection_publisher)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = InferenceYOLOv7()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()