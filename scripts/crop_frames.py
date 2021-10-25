"""
TODO crop faces only
"""
import sys,os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import cv2

import numpy as np

from fire import Fire
from pathlib import Path
from tqdm.cli import tqdm

def crop_frames(video_folder, frame_folder, n_crops):

    for video_fn in tqdm(list(video_folder.glob("**/*.mp4")), desc="frame cropping"):
        video_frame_folder = frame_folder / video_fn.stem
        video_frame_folder.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_fn))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_jump = n_frames // (n_crops-1)


        frames_to_crop = list(range(0, n_frames, frame_jump))
        if len(frames_to_crop) < n_crops:
            frames_to_crop.append(n_frames-1)

        frame_count = 0
        for frame_id in frames_to_crop:
            cap.set(1, frame_id)
            _, frame = cap.read()
            cv2.imwrite(str(video_frame_folder / f"frame={frame_count}.png"), frame)
            frame_count += frame_jump

        cap.release()

def main():
    data_folder = Path("data")
    save_folder = data_folder / "frames"
    n_crops = 4

    crop_frames(data_folder / "test", save_folder / "test", n_crops)
    crop_frames(data_folder / "train", save_folder / "train", n_crops)

if __name__ == "__main__":
    Fire(main)

