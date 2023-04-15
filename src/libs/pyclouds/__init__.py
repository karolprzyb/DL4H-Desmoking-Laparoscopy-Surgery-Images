HasPyglet = True
HasPygame = True

try:
	import pyglet
except ImportError:
	HasPyglet = False

try:
	import pygame
except ImportError:
	HasPygame = False

if HasPyglet:
	print("Using Pyglet")
	import PygletCore
elif HasPygame:
	print("Using Pygame")
	from .PygameCore import *
else:
	print("Install Pyglet or Pygame to run")