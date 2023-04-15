import Config as Cfg
from Clouds import CloudChunk, CloudManager


import pygame

class CloudRenderer(CloudChunk):
	def __init__(self, X, Generator):
		super(CloudRenderer, self).__init__(X, Generator)

		self.Surface = pygame.Surface((Cfg.CloudWidth, Cfg.CloudHeight), pygame.SRCALPHA)
		self.Source = Generator.Source

	def GenerateFinshed(self):
		print("start gen finished cloudrend")
		super(CloudRenderer, self).GenerateFinshed()	
		print("converting cloudrend gen finished")
		PSize = Cfg.PixelSize
		for Pos, Colour in self.PCMap.items():
			X, Y = Pos
			R, G, B, A = Colour

			self.Surface.fill((R * 255, G * 255, B * 255, A * 255), (X - self.X, Y, PSize, PSize))

	def Draw(self, X):
		super(CloudRenderer, self).Draw(X)
		self.Source.blit(self.Surface, (self.X - X, 0))

		#print "Drawing ", self.X, "at",self.X - X

class CloudRenderManager(CloudManager):
	def __init__(self, Source):
		super(CloudRenderManager, self).__init__(CloudRenderer)
		self.Source = Source

start_num = int(input())
pygame.init()

class GameWindow(object):
	def __init__(self):
		self.Width = 640
		self.Height = 480

		self.Screen = pygame.display.set_mode((self.Width, self.Height), pygame.SRCALPHA)
		self.Clock = pygame.time.Clock()

		self.Running = True

		self.Clouds = CloudRenderManager(self.Screen)

		self.XPos = 0.0
		self.XChange = 2.0

		self.Black=(0,0,0)

	def Loop(self):
		while self.Running:
			self.Tick()

	def Tick(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				self.Running = False
				return

		self.Screen.fill(self.Black)
		X = int(self.XPos / Cfg.CloudWidth)
		print(self.XPos)
		for CloudX in range(X, X + int(round(self.XPos / Cfg.CloudWidth)) + 3):
			self.Clouds.GetObject(CloudX * Cfg.CloudWidth).Draw(self.XPos)

		self.Clouds.genPNG("nothing for now", start_num)
		
		pygame.display.flip()

		self.XPos += self.XChange
		self.Clock.tick(Cfg.Framerate)


G = GameWindow()
G.Loop()