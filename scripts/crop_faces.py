import sys,os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import cv2
import dlib
import numpy as np

from fire import Fire
from pathlib import Path
from tqdm.cli import tqdm

def crop_frames(detector, video_folder, frame_folder, n_crops):

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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(frame, 1)
            if len(dets) == 0:
                continue
            # consider only first face
            det = dets[0]

            if det.confidence < 0.5:
                continue

            minx = max(0, det.rect.left())
            miny = max(0, det.rect.top())
            maxx = min(frame.shape[0], det.rect.right())
            maxy = min(frame.shape[1], det.rect.bottom())
            # print(minx, miny, maxx, maxy)
            face_crop = frame[miny: maxy, minx: maxx]
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            # change '=' symbol in name due to a kaggle dataset nameing policy
            cv2.imwrite(str(video_frame_folder / f"frame={frame_count}.png"), face_crop)

            frame_count += frame_jump

        cap.release()

        # cap = cv2.VideoCapture(str(video_fn))
        # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # frame_count = 0
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if ret:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         dets = detector(frame, 1)
        #         if len(dets) == 0:
        #             continue
        #         # consider only first face
        #         det = dets[0]

        #         if det.confidence < 0.5:
        #             continue

        #         minx = max(0, det.rect.left())
        #         miny = min(frame.shape[1], det.rect.top())
        #         maxx = min(frame.shape[0], det.rect.right())
        #         maxy = max(0, det.rect.bottom())
        #         # print(minx, miny, maxx, maxy)
        #         face_crop = frame[miny: maxy, minx: maxx]
        #         face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        #         cv2.imwrite(str(video_frame_folder / f"frame={frame_count}.png"), face_crop)

        #         frame_count += 1
        #     else:
        #         break
        # cap.release()

def main():
    data_folder = Path("data")
    save_folder = data_folder / "faces"
    n_crops = 4


    detector = dlib.cnn_face_detection_model_v1("archive/mmod-human-face-detect.dat")

    crop_frames(detector, data_folder / "test", save_folder / "test", n_crops)
    crop_frames(detector, data_folder / "train", save_folder / "train", n_crops)

if __name__ == "__main__":
    Fire(main)

