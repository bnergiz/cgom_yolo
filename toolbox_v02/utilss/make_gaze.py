"""
ST: 'cGOM'
Bachmann David, Hess Stephan & Julian Wolf (SV)
pdz, ETH Zürich
2018

This file contains all functions to perform gaze inference.
"""

# Global imports
import os
import sys
import skimage
import skimage.io
import warnings
import argparse
import cv2
import numpy as np

from models import *  # set ONNX_EXPORT in models.py
from utilss.datasets import *
from utilss.utils import *

# Local imports
import utils

# Import YOLOv3 from parent directory
sys.path.append('..')
#from yolov3.cfg import model.cfg
#from yolov3 import models as modellib

# Suppress warnings
warnings.filterwarnings('ignore', message='Anti-aliasing will be enabled by default in skimage 0.15 to')

# Import default_configs
parser = argparse.ArgumentParser(description='Map the gaze coordinates to object names.')
parser.add_argument('--config', default='./default_configs/make_gaze.yaml')
c = parser.parse_args()
args = utils.read_config(c.config)

################################################################################## Detecter Function


def detect(save_img=False):

    dets = []
    pos = []

    dataset = LoadImages('/Users/berkn/Desktop/ETH/Master/Semester_2/Semester_Project/cgom_yolo/toolbox_v02/middle', img_size=imgsz)
    
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # print(img)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                print(det)
                dets.append(det)

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

    return dets

    print('Done. (%.3fs)' % (time.time() - t0))


################################################################################## Detecter Function

# Derived config class
#class GazeConfig(Config):
class GazeConfig():

    # Name
    NAME = "gaze"

    # Number of GPUs
    GPU_COUNT = 1

    # Number of images per GPU
    IMAGES_PER_GPU = 1

    # Detection confidence
    DETECTION_MIN_CONFIDENCE = args['detection_min_confidence']

def track_gaze(model, name):

    counter = 0

    # Directories
    VIDEO_PATH = os.path.join(args['video_dir'], name + '.avi')
    GAZE_PATH = os.path.join(args['gaze_dir'], name + '.txt')
    print(GAZE_PATH)
    print(VIDEO_PATH)
    IMAGE_DIR = os.path.join(args['image_output_dir'], name)
    FILE_DIR = args['file_output_dir']
    utils.make_path(IMAGE_DIR)
    utils.make_path(FILE_DIR)

    # Video capture
    video = cv2.VideoCapture(VIDEO_PATH)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Read the gaze file and make it iterable
    gaze = utils.read_gaze(GAZE_PATH, max_res=(height, width))
    gaze = iter(gaze)

    # Set up an info buffer
    info_buffer = []

    # Get the first entry from the gaze file
    gaze_entry = next(gaze)
    (t_start, t_end, x, y) = list(gaze_entry.values())

    # If num_f is selected to be zero we choose the central frame from the fixation, else random frames are taken
    f_start = int(t_start * fps)
    f_end = int(t_end * fps)
    if args['num_f'] == 0:
        f_rand = [int((f_end + f_start + 2) / 2)]
    # Get random frames from the fixation interval
    # Shift the range, since np.arange takes a lower half open interval
    # We require this since int() crops the values and we use the frame count instead of the ID
    # Like this the lowest "frame ID" is 1, however, the lowest frame would be zero, since eg. >int(0.7) 0.
    else:
        f_range = np.arange(f_start, f_end) + 1
        rand_size = np.minimum(f_range.size, args['num_f'])
        # It might happen that both time steps lie within the same frame, hence the array would be of size 0
        # If that is the case, use f_start
        if f_range.size > 0:
            f_rand = np.random.choice(f_range, rand_size, replace=False)
        else:
            f_rand = [f_start + 1]

    # Go through the frames of the video
    f_count = 0
    video_flag = True
#    score_tracker = np.zeros(len(classes))
    while video_flag:

        # Read a frame
        video_flag, frame = video.read()
        f_count += 1

        # Just to be sure
       ### assert f_count <= max(f_rand), 'For some reason the frame counter surpassed all values in the random array.'

        # If a frame is within our random array we keep it
        if f_count in f_rand:
            assert video_flag, 'A time outside the scope of the video has been selected. This should not happen.'

            # Perform detection on the frame
            # Don't forget to flip it, due to open cv
            
            frame = np.flip(frame, axis=2).copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('/Users/berkn/Desktop/ETH/Master/Semester_2/Semester_Project/cgom_yolo/toolbox_v02/middle/image.jpg', frame)
            
#            r = model.detect([frame], verbose=0)[0]

            with torch.no_grad():
                dets = detect()

            counter = counter + 1
            label = 'BG'
            print(counter)
            

            # Change to YOLOv3
            # Change to YOLOv3
            # Change to YOLOv3
            # Change to YOLOv3


            # Check where the gaze point lies
            if len(dets) == 0:
                label = 'BG'
            else:
                dets_2 = dets[0]
                for l in range(6):
                    if len(dets_2) == 0:
                        break
                    else:
                        vec = dets_2[0]
                        x_1 = int(vec[0])
                        x_2 = int(vec[2])
                        y_1 = int(vec[1])
                        y_2 = int(vec[3])
                        print(x,y)
                        print(x_1,x_2,y_1,y_2)
                        if x > x_1 and x < x_2 and y > y_1 and y < y_2:
                            print('yes')
                            label = vec[5]               
                        dets_2 = dets_2[1:]                

            # If the fixation is exhausted, generate the required outputs and get new frame IDs
            # This is the case if the counter has reached the largest frame ID in the random array
            if f_count == max(f_rand):

                if len(dets) == 0:
                    label = 'BG'
                else:
                    if label == 0:
                        label = 'module'
                    if label == 1:
                        label = 'BG'
                    if label == 2:
                        label = 'tool'

                # Save the info for later
                info_buffer.append({'start_time': t_start, 'end_time': t_end, 'label':label})
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Write the latest frame to an image if you please
                if args['image_output_dir'] != None:
                    cv2.circle(frame, (x, y), int(frame.shape[0]/100.), (255, 0, 0), thickness=int(frame.shape[0]/100.))
                    text = 'start_time: %f end_time: %f label: %s' %(t_start, t_end, label)
                    cv2.putText(frame, text, (int(frame.shape[0]/50.), int(frame.shape[0]/50.)), cv2.FONT_HERSHEY_PLAIN, frame.shape[0]/700., (255, 0, 0))
                    skimage.io.imsave(os.path.join(IMAGE_DIR, str(f_count) + '.JPG'), frame)

                # Select a new entry from the gaze file, if it is exhausted break the loop
                gaze_entry = next(gaze, 'break')
                if gaze_entry == 'break':
                    break
                (t_start, t_end, x, y) = list(gaze_entry.values())

                # Either get the central frame or random frames
                f_start = int(t_start * fps)
                f_end = int(t_end * fps)
                if args['num_f'] == 0:
                    f_rand = [int((f_end + f_start + 2) / 2)]
                else:
                    # Get random frames from the fixation interval
                    f_range = np.arange(f_start, f_end) + 1
                    rand_size = np.minimum(f_range.size, args['num_f'])
                    if f_range.size > 0:
                        f_rand = np.random.choice(f_range, rand_size, replace=False)
                    else:
                        f_rand = [f_start + 1]

    video.release()

    # Write the info buffer to a file
    utils.write_gaze(info_buffer, os.path.join(FILE_DIR, name + '.txt'))


if __name__ == '__main__':

    # Set arguments for YOLOv3
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/model.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/hilti.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/weights.sh', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples/hilti', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    opt.names = list(glob.iglob('./**/' + opt.names, recursive=True))[0]  # find file
    print(opt)

    ################# Initialize

    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/weights.sh', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    ################# Initialize

    # Iterate through all videos and gaze files
    gaze_names = [name.split('.')[0] for name in os.listdir(args['gaze_dir'])]
    for video_name in os.listdir(args['video_dir']):
        # Detach name from suffix
        name = video_name.split('.')[0]
        # Check if corresponding gazefile exists
        assert name in gaze_names, "Corresponding gaze file does not exists."
        # Run the tracker function
        track_gaze(model, name)

