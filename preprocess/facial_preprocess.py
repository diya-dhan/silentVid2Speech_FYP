from os import listdir, path

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
import tensorflow as tf
from glob import glob


import face_detection

parser = argparse.ArgumentParser()

parser.add_argument("--speaker_root", help="Root folder of Speaker", required=True)

parser.add_argument("--speaker", help="Helps in preprocessing", required=True)


args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]
# face_detection.LandmarksType._2D allows to detect points in a 2D space
# 

def process_video_file(vfile):
	video_stream = cv2.VideoCapture(vfile)  #vfile is the video file which is being read
	
	frames = []	#gets the individual frames of the video and stores it in the frames array
	while 1:
		still_reading, frame = video_stream.read() #.read() helps read each frame
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)

	fulldir = vfile.replace('/sample/', '/preprocessed/')
	fulldir = fulldir[:fulldir.rfind('.')] # ignore extension
	
	os.makedirs(fulldir, exist_ok=True)
	
	batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

	i = -1
	for fb in batches:
		preds = fa.get_detections_for_batch(np.asarray(fb))

		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue

			cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), f[0])

def main(args):

	filelist = glob(path.join(args.speaker_root, '/*/*.mp4'))

	for i, vfile in enumerate(filelist):
		process_video_file(vfile)

if __name__ == '__main__':
	main(args)
