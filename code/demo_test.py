'''
	A demo for how to use FaceVerification function
	figure must be jpeg format
'''
from __future__ import absolute_import
from __future__ import print_function

from scipy import misc
from FaceVerification import FaceVerification
path1 = 'path/to/fig1'
path2 = 'path/to/fig2'
x1 = misc.imread(path1)
x2 = misc.imread(path2)
y = FaceVerification(x1,x2)
print(y)