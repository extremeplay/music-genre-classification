import os
import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt

__author__ = 'alex'
DATA_PATH = '/host/master-ACS/res/data'
GTZAN_PATH = DATA_PATH + '/genres'
GTZAN_POST_PATH = DATA_PATH + '/genres-post'
FEAT_DATA_PATH = GTZAN_POST_PATH + '/features'
CLASSIF_DATA_PATH = GTZAN_POST_PATH + '/classifiers'
TEST_RES_PATH = GTZAN_POST_PATH + '/tests'
OUTPUT_PATH = GTZAN_POST_PATH + '/output'

SKIP_FEAT_EXT = False
SKIP_TRAIN = False
SKIP_TEST = False
SKIP_PRES = False

def c2plot(data):
    plt.plot(range(200),data[:200])
    plt.show()

# preprocessing

# feature extractionr
if(SKIP_FEAT_EXT):
    pass
else:
    # genredirs = os.listdir(GTZAN_PATH)
    genredirs = ['jazz']
    for label,dirname in enumerate(genredirs):
        # files = os.listdir(GTZAN_PATH + '/' + dirname)
        files = ['jazz.00012.au.wav']
        for filename in files:
            file = GTZAN_PATH + '/' + dirname + '/' + filename
            [rate,data] = scipy.io.wavfile.read(file)
            c2plot(data)
            pass
# training

if(SKIP_TRAIN):
    pass
else:
    pass

# testing
if(SKIP_TRAIN):
    pass
else:
    pass

# results presentation
if(SKIP_TRAIN):
    pass
else:
    pass
pass