"""
ST: 'cGOM'
Bachmann David, Hess Stephan & Julian Wolf (SV)
pdz, ETH Zürich
2018

This file contains all functions to extract images from video, also just from fixations.
"""

# Global imports
import cv2
import argparse
import random
import skimage.io
import os
import numpy as np

# Local imports
from utils import read_gaze


def extract_images_from_video(args):

    # Read video
    video = cv2.VideoCapture(args.video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Either no gaze file is provided, then each frame is chosen at random from the entire video
    if args.gaze_path == None:

        # Calculate the probabilities in order to end up with config.num_images
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        p = args.num_images / num_frames

        # Write the corresponding frames
        i = 0
        video_flag = True
        while video_flag:
            video_flag, frame = video.read()
            if random.random() < p:
                skimage.io.imsave(os.path.join(args.output_dir, 'image_' + str(i) + '.JPG'), np.flip(frame, axis=2))
                i += 1

    # If a gaze file is provided, we only pull a frame from fixations
    else:

        # Read the gaze file and make it iterable
        gaze = read_gaze(args.gaze_path, max_res=(height, width))
#        print((gaze))
        p = args.num_images / len(gaze)
        gaze = iter(gaze)

        # Random frame
        gaze_entry = next(gaze)
        (t_start, t_end, x, y) = list(gaze_entry.values())
        rand_frame = np.random.uniform(t_start, t_end) * fps
        rand_frame = int(rand_frame)

        i = 0
        f_count = 0
        video_flag = True
        while video_flag:
            video_flag, frame = video.read()
            f_count += 1

            # If the random frame corresponds to the frame ID we inspect it
            if rand_frame == f_count:

                # However, the frame is only kept with probability p
                if random.random() < p:
                    skimage.io.imsave(os.path.join(args.output_dir, 'image_' + str(i) + '.JPG'), np.flip(frame, axis=2))
                    i += 1

                # Get next random frame
                gaze_entry = next(gaze)
                (t_start, t_end, x, y) = list(gaze_entry.values())
                rand_frame = np.random.uniform(t_start, t_end) * fps
                rand_frame = int(rand_frame)

    video.release()

if __name__ == '__main__':

    # Get config
    parser = argparse.ArgumentParser(description='Extract a number of random frames from a video')
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_images', type=int, required=True)
    parser.add_argument('--gaze_path', default=None)
    args = parser.parse_args()

    # Extract frames
    extract_images_from_video(args)
