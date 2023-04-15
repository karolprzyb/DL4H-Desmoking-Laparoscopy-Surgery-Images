from math import cos, pi
from noise import SimplexNoiseGen as noisy
from threading import Thread
import queue as Queue
from time import time
from time import sleep
import Config
import cv2
import numpy as np

class ObjectManager(object):
	def __init__(self, ObjectClass):
		self.Object = {}
		self.ObjectClass = ObjectClass

	def GetObject(self, X):
		if X in self.Object:
			return self.Object[X]
		Obj = self.ObjectClass(X, self)
		self.Object[X] = Obj

		return Obj
	
	def genPNG(self, filename:str, startNum:int=0):
		# Just make a PNG of the cloud we want to generate, forget the rest.
		img = np.empty((Config.CloudWidth, Config.CloudHeight, 4))
		for x in range(0,250):
			print("Creating png " + str(startNum+x))
			self.Noise.resf()
			Obj = self.ObjectClass(100, self)
			while(Obj.Finished is False):
				sleep(0.5)
			img = np.array(Obj.Colours)*255.0
			img = img.astype(np.uint8)
			img = img.reshape((int(Config.CloudWidth/Config.PixelSize), int(Config.CloudHeight/Config.PixelSize), 4))
			isg = cv2.imwrite("C:/Users/Karol/Documents/DL4H/test2/smoke_" + str(startNum+x) + ".png", img)
			print(isg)

	def genPNGParallel(self, filename:str):
		#THIS IS NOT ACTUALLY FASTER BECAUSE OF GIL.... forgot about that...
		# Just make a PNG of the cloud we want to generate, forget the rest.
		img = np.empty((Config.CloudWidth, Config.CloudHeight, 4))
		numCores = 16
		pendingObjs = []
		pendingRans = [None]*numCores
		pendingRemoval = []
		numPNGs = 0
		targetPNGs = 1000
		exRandom = random.Random(int(time()))
		for x in range(numCores):
			pendingRans[x] = dummyGen(noisy(time()+200.0*exRandom.random()))
		while numPNGs < targetPNGs:
			#self.Noise.resf()
			#Obj = self.ObjectClass(100, self)
			noPass = True
			while(len(pendingObjs) < numCores):
				pendingObjs.append(self.ObjectClass(100, self))
			for p_i, pending in enumerate(pendingObjs):
				if pending.Finished is True:
					img = np.array(pending.Colours)*255.0
					img = img.astype(np.uint8)
					img = img.reshape((int(Config.CloudWidth/Config.PixelSize), int(Config.CloudHeight/Config.PixelSize), 4))
					isg = cv2.imwrite("C:/Users/Karol/Documents/DL4H/test/" + filename + "_" + str(numPNGs) + ".png", img)
					print("PNG ", str(numPNGs), " Complete")
					print(isg)
					pendingRemoval.append(p_i)
					numPNGs += 1
			for rem in sorted(pendingRemoval, reverse=True):
				pendingObjs.pop(rem)
				pendingRans[rem].Noise.resf()
				



class dummyGen(object):
	def __init__(self, noiseIn):
		self.Noise = noiseIn

      

class CloudManager(ObjectManager):
	def __init__(self, CloudClass = None):
		self.Noise = noisy(Config.Seed)

		if CloudClass == None:
			CloudClass = CloudChunk
		super(CloudManager, self).__init__(CloudClass)


class CloudChunk(object):
	def __init__(self, XPos, Generator):
		self.X = XPos
		self.Noise = Generator.Noise
		self.Generator = Generator

		self.Finished = False

		T = Thread(target=self.Generate)
		T.daemon = True
		T.start()

	def Generate(self):
		print("Starting Generation at",self.X)
		start = time()
		Points = []
		Colours = []
		Length = 0

		PCMap = {}

		#Generation stuff
		PixelSize = Config.PixelSize

		YOffset = Config.CloudHeight / 2.0

		Noise = self.Noise
		NoiseOffset = Config.NoiseOffset

		for X in range(0, Config.CloudWidth - 1, PixelSize):
			XOff = X+self.X

			for Y in range(0, Config.CloudHeight - 1, PixelSize):
				Points.append(XOff)
				Points.append(Y)

				Colours.append(1)
				Colours.append(1)
				Colours.append(1)

				#Get noise, round and clamp
				NoiseGen = Noise.fBm(XOff, Y) + NoiseOffset
				NoiseGen = max(0, min(1, NoiseGen))
				
				# Fade around the edges - use cos to get better fading
				Diff = abs(Y - YOffset) / YOffset
				NoiseGen *= cos(Diff * pi / 2)
				
				Colours.append(NoiseGen)

				if NoiseGen > 0:
					PCMap[(XOff, Y)] = (1, 1, 1, NoiseGen)

				Length += 1

		#Assign variables
		self.Points = Points
		self.Colours = Colours
		self.Length = Length
		self.PCMap = PCMap

		print("Finished Generation at", self.X)
		print("\tTook",time() - start)
		self.Finished = True

	def GenerateFinshed(self):
		print("running super now")
		pass
		

	def Draw(self, X):
		if self.Finished:
			self.Finished = False
			print("draw triggered cloudchunk")
			self.GenerateFinshed()
		else:
			return False
