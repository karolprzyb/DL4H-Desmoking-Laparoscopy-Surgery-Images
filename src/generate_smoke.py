#import libs.pyclouds.PygameCore as pyclouds
import cv2

class inside(object):
    def __init__(self, testint):
        self.num = testint
        self.addten()
    def addten(self):
        self.num+=10
    def addme(self):
        print("inside")
    def testme(self):
        self.addme()

class outside(inside):
    def __init__(self, testint):
        super().__init__(testint)
    def addten(self):
        print("this is 10")
        super().testme()
    def addme(self):
        print("outside")
        

test = outside(14)
#test.addten()

print(test.num)