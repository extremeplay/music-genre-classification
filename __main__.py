__author__ = 'alex'

import numpy
import glob
import os
import scipy.stats
import scipy.io.wavfile
import matplotlib.pyplot
from scikits.talkbox.features import mfcc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


DATA_PATH = '/host/master-ACS/res/data'
GTZAN_PATH = DATA_PATH + '/genres'
GTZAN_POST_PATH = DATA_PATH + '/genres-post'
FEAT_DATA_PATH = GTZAN_POST_PATH + '/features'
CLASSIF_DATA_PATH = GTZAN_POST_PATH + '/classifiers'
TEST_EVAL_PATH = GTZAN_POST_PATH + '/evaluation'
GRAPH_PATH = GTZAN_POST_PATH + '/graphs'

SKIP_FEAT_EXT = True
SKIP_TRAIN = False
SKIP_TEST = False
SKIP_GRAPH = False


def c2plot(d):
    matplotlib.pyplot.plot(range(200), d[:200])
    matplotlib.pyplot.show()


def c2mean(dat):
    meanvec = [0.0] * len(dat[0])
    lenvec = [0] * len(dat[0])
    for arr in dat:
        # if not (any(numpy.isnan(arr)) or any(numpy.isinf(arr))):
        for i in range(len(arr)):
            lenvec[i] += 1
            meanvec[i] += arr[i]
    return [x / y for (x, y) in zip(meanvec, lenvec)]


def extractfeatures():
    genredirs = sorted(os.listdir(GTZAN_PATH))
    for dirname in genredirs:
        files = sorted(glob.glob(GTZAN_PATH + '/' + dirname + '/' + '*.wav'))
        for filename in files:
            f = filename
            [_, data] = scipy.io.wavfile.read(f)
            ceps, _, _ = mfcc(data)
            fceps = FEAT_DATA_PATH + '/' + dirname + '/' + os.path.basename(filename) + '.ceps'
            numpy.save(fceps, ceps)


def extractfeatures2():
    genredirs = sorted(os.listdir(FEAT_DATA_PATH))
    for dirname in genredirs:
        files = sorted(glob.glob(FEAT_DATA_PATH + '/' + dirname + '/' + '*.ceps.npy'))
        for filename in files:
            ceps = numpy.load(filename)
            fceps = FEAT_DATA_PATH + '/' + dirname + '/' + os.path.basename(filename) + '.summ'
            numpy.save(fceps, scipy.stats.nanmedian(ceps,axis=0))


def loadfeatures():
    xlocal = []
    ylocal = []
    songslocal = []
    genredirs = sorted(os.listdir(FEAT_DATA_PATH))
    for label, dirname in enumerate(genredirs):
        files = sorted(glob.glob(FEAT_DATA_PATH + '/' + dirname + '/' + '*.summ.npy'))
        for f in files:
            summ = numpy.load(f)
            # ceps = [arr for arr in ceps if not (any(numpy.isinf(arr)) or any(numpy.isnan(arr)))]
            xlocal.append(summ)
            ylocal.append(label)
            songslocal.append(f)
    return xlocal, ylocal, songslocal

# preprocessing
# convert .au to .wav using bash

# feature extractionr
if SKIP_FEAT_EXT:
    pass
else:
    extractfeatures()

# load features
X, y, songs = loadfeatures()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# training

if SKIP_TRAIN:
    pass
else:
    clsf = LogisticRegression()
    clsf.fit(X_train, y_train)
    cfile = CLASSIF_DATA_PATH + '/' + 'logistic.pkl'
    _ = joblib.dump(clsf, cfile, compress=9)


# load classifier

cfile = CLASSIF_DATA_PATH + '/' + 'logistic.pkl'
clsf = joblib.load(cfile)


# testing
if SKIP_TEST:
    pass
else:
    pass

# load test results


# results presentation
if SKIP_GRAPH:
    pass
else:
    pass
pass