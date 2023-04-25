# Copyright (c) 2023, Karol Przybsyzewski - karolprzyb @github
# Under MIT License

import noise
import Config
import cv2
import numpy as np
import time
import math
import argparse
import sys
from threading import Thread
from subprocess import Popen
import signal

class SmokeImg(object):
	def __init__(self, Generator, filepath, fileprefix, startNum=0):
		self.noise = Generator
		self.width = math.floor(Config.CloudWidth/Config.PixelSize)
		self.height = math.floor(Config.CloudHeight/Config.PixelSize)
		self.BGRAimg = np.empty((self.width, self.height, 4))
		self.number = 0
		self.filepath = filepath
		self.fileprefix = fileprefix
		self.startNum = startNum

	def Generate(self):
		print("Starting Image Generation for Image " + str(self.number + self.startNum))
		start = time.time()

		#Generation stuff
		PixelSize = Config.PixelSize

		YOffset = Config.CloudHeight / 2.0

		NoiseOffset = Config.NoiseOffset
		rX, rY = 0, 0

		for X in range(0, Config.CloudWidth - 1, PixelSize):
			for Y in range(0, Config.CloudHeight - 1, PixelSize):
				#Get noise, round and clamp
				NoiseGen = self.noise.fBm(X, Y)
				NoiseGen = max(0, min(1, NoiseGen))
				
				# Fade around the edges - use cos to get better fading
				Diff = abs(Y - YOffset) / YOffset
				NoiseGen *= 255.0*math.cos(Diff * math.pi / 2)
    
				self.BGRAimg[rX,rY,:] = [255.0,255.0,255.0,NoiseGen]
				rY += 1
			rY = 0
			rX += 1

		print("Finished Generation")
		print("\tTook",time.time() - start)

	def writePNG(self):
		print("Writing image to disk")
		start = time.time()
		cv2.imwrite(self.filepath + self.fileprefix + "_" + str(self.startNum + self.number) + ".png", self.BGRAimg)
		self.number += 1
		print("Finished Writing")
		print("\tTook",time.time() - start)

def reconcileArgs(args):
    if(args.end == 0):
        print("NO END INDEX GIVEN. SETTING TO numVariations IN CONFIG: " + str(Config.numVariations))
        args.end = Config.numVariations
    if(args.bdir == ''):
        print("No base directory given. Using: " + Config.baseDir)
        args.bdir = Config.baseDir
    if(args.bfname == ''):
        print("No base filename given, using: " + Config.baseFileName)
        args.bfname = Config.baseFileName

def runWorker(args):
	print("Running worker " + str(args.wID))
	generator = noise.SimplexNoiseGen(time.time())
	smoker = SmokeImg(generator, args.bdir, args.bfname, args.start)
	for x in range(args.start, args.end):
		smoker.Generate()
		smoker.writePNG()
		smoker.noise.resf()
	print("Worker done, ID: " + str(args.wID))

def initParser():
	parser = argparse.ArgumentParser(description="Fake smoke generator")

	parser.add_argument('--worker', action = "store_true", default=False)
	parser.add_argument('--start', type=int, metavar='start_index', default=0)
	parser.add_argument('--end', type=int, metavar='end_index', default=0)
	parser.add_argument('--nworkers', type=int, metavar='number_of_workers', default=1)
	parser.add_argument('--bdir', type=str, metavar='base_directory', default='')
	parser.add_argument('--bfname', type=str, metavar='base_file_name', default='')
	parser.add_argument('--wID', type=int, metavar='worker_ID', default=0)

	return parser

def callWorker(args, startNum, endNum, ID):
	if sys.executable == '':
		Exception("CANNOT GET PYTHON EXECUTABLE PATH - EXITING")
		return
	command = sys.executable + ' ' +  sys.argv[0] + ' --worker'
	command += ' --start ' + str(startNum) + ' --end ' + str(endNum)
	command += ' --bdir ' + args.bdir + ' --bfname ' + args.bfname
	command += ' --wID ' + str(ID)
	print(command)
	runner = None
	try:
		runner = Popen(command, shell=True)
		print("Launched worker: " + str(ID))
		while runner.poll() is None:
			time.sleep(1.0)
	except KeyboardInterrupt:
		print("Exiting, ctrl+c")
		runner.send_signal(signal.CTRL_C_EVENT)
		sys.exit()

	return

# def runAsFunction(baseDir='', end = Config.numVariations, baseFileName='', nworkers = Config.numWorkers):
# 	parser = initParser()
# 	args = parser.parse_args('')
# 	reconcileArgs(args)
# 	args.bdir = baseDir
# 	args.end = end
# 	args.bfname = baseFileName
# 	args.nworkers = nworkers

# 	print("IDENTIFY AS WORKER: " + str(args.worker))

# 	if args.worker:
# 		try:
# 			runWorker(args)
# 		except KeyboardInterrupt:
# 			print("Exiting, ctrl+c")
# 			sys.exit()
# 	else:
# 		print("Starting workers")
# 		threads  = []
# 		perWorker = math.ceil((args.end-args.start)/args.nworkers)
# 		print("Starting " + str(args.nworkers) + " workers...")
# 		try:
# 			for x in range(0, args.nworkers):
# 				threads.append(Thread(target=callWorker, args=[args, perWorker*x, max(perWorker*(x+1),args.end), x]))
# 				threads[x].start()
# 			for x in threads:
# 				while(x.is_alive()):
# 					time.sleep(1.0)
# 			print("Workers all finished")
# 		except KeyboardInterrupt:
# 			print("Exiting, ctrl+c")

if __name__ == '__main__':
	parser = initParser()
	args = parser.parse_args()
	reconcileArgs(args)

	print("IDENTIFY AS WORKER: " + str(args.worker))

	if args.worker:
		try:
			runWorker(args)
		except KeyboardInterrupt:
			print("Exiting, ctrl+c")
			sys.exit()
	else:
		print("Starting workers")
		threads  = []
		perWorker = math.ceil((args.end-args.start)/args.nworkers)
		print("Starting " + str(args.nworkers) + " workers...")
		try:
			for x in range(0, args.nworkers):
				threads.append(Thread(target=callWorker, args=[args, perWorker*x, min(perWorker*(x+1),args.end), x]))
				threads[x].start()
			for x in threads:
				while(x.is_alive()):
					time.sleep(1.0)
			print("Workers all finished")
		except KeyboardInterrupt:
			print("Exiting, ctrl+c")
