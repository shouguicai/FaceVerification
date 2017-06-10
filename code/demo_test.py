'''
	A demo for how to use FaceVerification function
'''
from __future__ import absolute_import
from __future__ import print_function

from scipy import misc
from FaceVerification import FaceVerification
path1 = './datasets/my_dataset/test/0000045/013.jpg'
path2 = './datasets/my_dataset/test/0000045/015.jpg'
x1 = misc.imread(path1)
x2 = misc.imread(path2)
y = FaceVerification(x1,x2)
print(y)