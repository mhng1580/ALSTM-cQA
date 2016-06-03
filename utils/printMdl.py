from __future__ import print_function
import numpy
import sys
import os

mdl = numpy.load(sys.argv[1])
print('\n')
print('Dumping model information:\n{0}\n'.format(os.path.basename(sys.argv[1])))
print('Hyper Parameters:\n{0}\n'.format(mdl['hypParams']))
print('Epoch Error:\n{0}\n'.format(mdl['epochError']))
print('Best Validation Error:\nIndex: {0}\tError: {1}\n'.format(numpy.argmin(mdl['histError'][:,4]), numpy.min(mdl['histError'][:,4])))
print('Best Test Error:\nIndex: {0}\tError: {1}\n'.format(numpy.argmin(mdl['histError'][:,5]), numpy.min(mdl['histError'][:,5])))
bestValIdx = numpy.argmin(mdl['histError'][:,4])
print('Test Error on Best Validation Index:\n{0}\n'.format(mdl['histError'][bestValIdx,5]))
