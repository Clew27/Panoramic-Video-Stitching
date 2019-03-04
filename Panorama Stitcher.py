#### IMPORTANT
#### Works under the assumption that the frame is 1920 x 1080 and the audio sampling rate is 48000 hz

import cv2 as cv
import numpy as np
import os
import subprocess as sp
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from time import sleep
from Stitcher import Stitcher

### CONSTANTS

## FILE CONSTANTS
FFMPEG_BIN         = 'ffmpeg'
FILE_DIRECTORY     = './Pano Footage/'
CAMERA_DIRECTORIES = ['LEFT', 'CENTER', 'RIGHT']
CURRENT_DIRECTORY  = os.getcwd()
VIDEO_NAME         = '3'

## SYNCING CONSTANTS
SAMPLE_RATE    = 48000 # hz
SLICE_SIZE     = int(1        * SAMPLE_RATE)
SLICE_INC      = int(1 / 1920 * SAMPLE_RATE)
MAX_TIME       = int(7.5      * SAMPLE_RATE)
CLAP_THERSHOLD = 20000 # intensity

## LOWPASS CONSTANTS
CUT_OFF = 1500 # hz
ORDER   = 15

## STITCHING CONSTANTS
OVERLAP         = 10 # %


### FUNCTIONS

def lowpass_filter(data, cut_off, freq, order = 4):
    nyq          = 0.5 * freq
    normalCutoff = cut_off / nyq
    b, a         = butter(order, normalCutoff, btype='lowpass')
    return lfilter(b, a, data)

def convert_audio_files():
    audio_files = list()
    for camera_dir in CAMERA_DIRECTORIES:
        audio_filename = os.path.join(CURRENT_DIRECTORY, FILE_DIRECTORY, camera_dir, '{}.wav'.format(VIDEO_NAME))
        audio_files.append(audio_filename)
        try:
            os.remove(audio_filename)  # Delete old file if possible
        except:
            pass
        split_audio_command = [FFMPEG_BIN,
                               '-i', audio_filename.replace('.wav', '.mp4'),
                               '-ac', '1',
                               audio_filename]
        sp.Popen(split_audio_command)

    sleep(1)  # Allow ffmpeg to run

    return audio_files

def get_audio(audio_files):
    audio_data = list()
    for audio_file in audio_files:
        _, data = wavfile.read(audio_file)
        data = lowpass_filter(data, CUT_OFF, SAMPLE_RATE, ORDER)  # Remove noise
        audio_data.append(np.abs(data))

    return audio_data

def get_clap_idxs():
    audio_files = convert_audio_files()
    audio_data = get_audio(audio_files)

    clap_idx = np.where(audio_data[2] >= CLAP_THERSHOLD)[0][0]  # Assuming the right camera is the last to start
    clap_slice = audio_data[2][clap_idx:clap_idx + SLICE_SIZE]

    np.seterr(all='raise')
    correlations = [list() for i in range(2)]
    for idx in range(0, MAX_TIME, SLICE_INC):
        for camera in range(2):
            try:
                corr = np.corrcoef(audio_data[camera][idx:idx + SLICE_SIZE], clap_slice)[0][1]
                correlations[camera].append(corr ** 2)
            except:
                correlations[camera].append(0)

    correlations = [np.nan_to_num(corr) for corr in correlations]

    clap_idxs = list()
    for camera in range(2):
        max = np.argmax(correlations[camera])
        max_sample = max * SLICE_INC  # The corresponding sample (convert corr_sample -> audio_sample)
        clap_idxs.append(max_sample)
    clap_idxs.append(clap_idx)

    return clap_idxs


### MAIN CODE

## AUDIO SYNCING

# SPLIT AUDIO
print("Syncing Video")

## FIND CORRELATION BASED ON CLAP
clap_idxs = get_clap_idxs()

## STITCH VIDEO
vid_caps = list()
for camera_dir in CAMERA_DIRECTORIES:
    cap = cv.VideoCapture(os.path.join(CURRENT_DIRECTORY, FILE_DIRECTORY, camera_dir, '{}.mp4'.format(VIDEO_NAME)))
    vid_caps.append(cap)

fps = vid_caps[0].get(cv.CAP_PROP_FPS)

# Remove frames so they all start at the hand clap
for camera, clap_idx in enumerate(clap_idxs):
    skip_frame_count = int(clap_idx / SAMPLE_RATE * fps)
    for i in range(skip_frame_count):
        vid_caps[camera].read()

print("Video successfully synced")

stitcher = Stitcher(OVERLAP)
# Frames of synchronized video (assuming the left is the first to stop recording)
stitcher.calibrate_stitcher(vid_caps)

#The rest of the video
print("Homography found... stitching rest of the video")

remaining_frames = int(vid_caps[0].get(cv.CAP_PROP_FRAME_COUNT)) - int(clap_idxs[0] / SAMPLE_RATE * fps) - 1

stitcher.stitch_video("Panoramic Video {}".format(VIDEO_NAME), vid_caps, fps, remaining_frames)

for i in range(3):
    vid_caps[i].release()