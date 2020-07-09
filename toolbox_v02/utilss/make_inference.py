"""
ST: 'cGOM'
Bachmann David, Hess Stephan & Julian Wolf (SV)
pdz, ETH Zürich
2018

This file contains all functions to perform inference on a single image.
"""

# Global imports
import os
import sys
import skimage
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

# Import default_configs
parser = argparse.ArgumentParser(description='Make simple predictions on one image.')
parser.add_argument('--config', default='./default_configs/make_inference.yaml')
c = parser.parse_args()
args = utils.read_config(c.config)

# Derived config class
class InferenceConfig(Config):

    # Name
    NAME = "inference"

    # Number of GPUs
    GPU_COUNT = 1

    # Number of images per GPU
    IMAGES_PER_GPU = 1

    # Detection confidence
    DETECTION_MIN_CONFIDENCE = args['detection_min_confidence']


if __name__ == '__main__':

    # Read the labels
    LABEL_DIR = os.path.join(args['label_dir'], 'labels.json')
    classes = list(utils.load(LABEL_DIR).keys())

    # Configs for model
    class InferenceConfig(InferenceConfig):
        NUM_CLASSES = len(classes)
    config = InferenceConfig()
    config.display()

    # Load the model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args['log_dir'])

    # Load weights
    if args['weights_path'] == 'last':
        model.load_weights(model.find_last(), by_name=True)
    else:
        model.load_weights(args['weights_path'], by_name=True)

    # Read image
    image = skimage.io.imread(args['image_path'])

    # Detect objects
    r = model.detect([image], verbose=1)[0]

    # Diliate the masks
    kernel = np.ones((args['dilation'], args['dilation']), dtype='uint8')
    masks = r['masks'].astype('uint8')
    dil_masks = np.zeros_like(masks)
    for i in range(masks.shape[-1]):
        dil_masks[:, :, i] = cv2.dilate(masks[:, :, i], kernel, iterations=1)

    # Visualize the image with all results
    visualize.display_instances(image, r['rois'], dil_masks, r['class_ids'], classes, r['scores'])
