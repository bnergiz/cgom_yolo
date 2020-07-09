"""
ST: 'cGOM'
Bachmann David, Hess Stephan & Julian Wolf (SV)
pdz, ETH Zürich
2018

This file contains all functions to create a gaze video.
"""

# Global imports
import os
import sys
import warnings
import argparse
import cv2
import numpy as np

# Local imports
import utils

# Import Mask RCNN from parent directory
sys.path.append('..')
from mrcnn.config import Config
from mrcnn import model as modellib, visualize

# Suppress warnings
warnings.filterwarnings('ignore', message='Anti-aliasing will be enabled by default in skimage 0.15 to')

# Derived config class
class GazeConfig(Config):

    # Name
    NAME = "gaze"

    # Number of GPUs
    GPU_COUNT = 1

    # Number of images per GPU
    IMAGES_PER_GPU = 1


def make_mask_gaze_video(model, args, classes):

    # Video capture
    video = cv2.VideoCapture(args.video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Video writer
    name = os.path.basename(args.video_path).split('.')[0]
    writer = cv2.VideoWriter(name + '_masked.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    # Read the gaze file and make it iterable
    gaze = utils.read_gaze(args.gaze_path, max_res=(height, width))
    gaze = iter(gaze)

    # Get the first entry from the gaze file
    gaze_entry = next(gaze)
    (t_start, t_end, x, y) = list(gaze_entry.values())
    f_start = int(t_start * fps)
    f_end = int(t_end * fps)

    # Go through the frames of the video
    f_count = 0
    video_flag = True
    while video_flag:

        # Read a frame
        video_flag, frame = video.read()
        f_count += 1

        # Detect masks
        frame = np.flip(frame, axis=2).copy()
        r = model.detect([frame], verbose=0)[0]
        masks = r['masks']
        scores = r['scores']
        class_ids = r['class_ids']
        num_masks = masks.shape[-1]

        # Apply the masks
        for i in range(num_masks):
            id = class_ids[i]
            color = (float(id == 1), float(id == 2), float(id > 2))
            frame = visualize.apply_mask(frame, masks[:, :, i], color)

        # If a frame is within our random array we keep it
        if f_start < f_count <= f_end:
            assert video_flag, 'A time outside the scope of the video has been selected. This should not happen.'

            # Check where the gaze point lies
            max_score = 0.
            max_class = 0
            for i in range(num_masks):
                if masks[y, x, i] == True and scores[i] > max_score:
                    max_score = scores[i]
                    max_class = class_ids[i]

            # Write the class onto the frame
            label = classes[max_class]
            text = 'label: %s' %label
            cv2.putText(frame, text, (int(frame.shape[0] / 50.), int(frame.shape[0] / 50.)), cv2.FONT_HERSHEY_PLAIN,
                        frame.shape[0] / 700., (255, 0, 0))

            # Draw the circle
            cv2.circle(frame, (x, y), int(frame.shape[0] / 100.), (255, 0, 0), thickness=int(frame.shape[0] / 100.))

            # Check if we are leaving the fixation
            # Exhaust casues the last frame not to be written - whatever.
            if f_count == f_end:
                gaze_entry = next(gaze, 'break')
                if gaze_entry == 'break':
                    break
                (t_start, t_end, x, y) = list(gaze_entry.values())
                f_start = int(t_start * fps)
                f_end = int(t_end * fps)


        # Write the frame
        writer.write(np.flip(frame, axis=2))

    writer.release()
    video.release()

if __name__ == '__main__':

    # Args
    parser = argparse.ArgumentParser(description='Make a fancy videos containing masks, gaze point, and detections')
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--gaze_path', required=True)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--weights_path', default='../weights/w_pilot.h5')
    parser.add_argument('--label_dir', default='../labels')
    parser.add_argument('--detection_min_confidence', default=0.9, type=int)
    args= parser.parse_args()

    # Read the labels
    LABEL_DIR = os.path.join(args.label_dir, 'labels.json')
    classes = list(utils.load(LABEL_DIR).keys())

    # Configs for model
    class GazeConfig(GazeConfig):
        NUM_CLASSES = len(classes)
        DETECTION_MIN_CONFIDENCE = args.detection_min_confidence
    config = GazeConfig()
    config.display()

    # Load the model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.log_dir)

    # Load weights
    if args.weights_path == 'last':
        model.load_weights(model.find_last(), by_name=True)
    else:
        model.load_weights(args.weights_path, by_name=True)

    # Make the video
    make_mask_gaze_video(model, args, classes)
