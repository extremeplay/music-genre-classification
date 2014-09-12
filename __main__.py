__author__ = 'alex'

import numpy
import glob
import os
import scipy.stats
import scipy.io.wavfile
import matplotlib.pyplot
from matplotlib import pyplot
from scikits.talkbox.features import mfcc
import sklearn.cross_validation
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


DATA_PATH = '/host/master-ACS/res/data'
GTZAN_PATH = DATA_PATH + '/genres'
GTZAN_POST_PATH = DATA_PATH + '/genres-post'
FEAT_DATA_PATH = GTZAN_POST_PATH + '/features'
CLASSIF_DATA_PATH = GTZAN_POST_PATH + '/classifiers'
TEST_EVAL_PATH = GTZAN_POST_PATH + '/evaluation'
GRAPH_PATH = GTZAN_POST_PATH + '/graphs'

FOLDS = 10

SKIP_FEAT_EXT = True
SKIP_TRAIN = False
# SKIP_TEST = True
SKIP_GRAPH = False
SKIP_CROSS_VALIDATE = False


def c2plot_roc(fpr, tpr, title, filename):
    pyplot.clf()
    pyplot.figure(num=None, figsize=(5, 4))
    pyplot.grid(True)
    pyplot.plot([0, 1], [0, 1], 'k--')
    pyplot.plot(fpr, tpr)
    pyplot.fill_between(fpr, tpr, alpha=0.5)
    pyplot.xlim([-0.1, 1.1])
    pyplot.ylim([-0.1, 1.1])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title(title, verticalalignment="bottom")
    pyplot.legend(loc="lower right")
    pyplot.savefig(os.path.join(GRAPH_PATH, "roc_" + filename + ".png"), bbox_inches="tight")


def c2plot_prc(recall, precision, title, name):
    pyplot.clf()
    pyplot.figure(num=None, figsize=(5, 4))
    pyplot.grid(True)
    pyplot.plot([0, 1], [1, 0], 'k--')
    pyplot.fill_between(recall, precision, alpha=0.5)
    pyplot.plot(recall, precision)
    pyplot.xlim([-0.1, 1.1])
    pyplot.ylim([-0.1, 1.1])
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.title(title)
    pyplot.savefig(os.path.join(GRAPH_PATH, "pr_" + name + ".png"), bbox_inches="tight")


def c2plot_confmat(cm, genres, title):
    from matplotlib import rcParams

    rcParams.update({'figure.autolayout': True})
    pyplot.clf()
    pyplot.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pyplot.axes()
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=45)
    # ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.grid(False)
    pyplot.xlabel('Predicted class')
    pyplot.ylabel('True class')
    pyplot.grid(False)
    pyplot.savefig(os.path.join(GRAPH_PATH, "confusion_matrix.png"), bbox_inches="tight")


def c2plot(d):
    matplotlib.pyplot.plot(range(200), d[:200])
    matplotlib.pyplot.show()


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
            numpy.save(fceps, scipy.stats.nanmedian(ceps, axis=0))


def loadfeatures():
    Xlocal = []
    ylocal = []
    songslocal = []
    genrelocal = []
    genredirs = sorted(os.listdir(FEAT_DATA_PATH))
    for label, dirname in enumerate(genredirs):
        files = sorted(glob.glob(FEAT_DATA_PATH + '/' + dirname + '/' + '*.summ.npy'))
        for f in files:
            summ = numpy.load(f)
            # ceps = [arr for arr in ceps if not (any(numpy.isinf(arr)) or any(numpy.isnan(arr)))]
            Xlocal.append(summ)
            ylocal.append(label)
            songslocal.append(f)
        genrelocal.append(dirname)
    return numpy.array(Xlocal), numpy.array(ylocal), numpy.array(songslocal), numpy.array(genrelocal)

# preprocessing
# convert .au to .wav using bash

# feature extractionr
if SKIP_FEAT_EXT:
    pass
else:
    extractfeatures()
    extractfeatures2()

# load features
X, y, songs, genres = loadfeatures()

score_all = []
cm_all = []
prcurve_p_all = []
prcurve_r_all = []
prcurve_thresh_all = []
prcurve_score_all = []
roc_tpr_all = []
roc_fpr_all = []
roc_thresh_all = []
roc_score_all = []

if SKIP_TRAIN:
    pass
else:
    splits = []
    if SKIP_CROSS_VALIDATE:
        train_idx, test_idx = sklearn.cross_validation.train_test_split(range(len(X)), train_size=0.8)
        splits.append((train_idx, test_idx))
    else:
        for train_idx, test_idx in sklearn.cross_validation.KFold(n=len(X), n_folds=FOLDS, shuffle=True):
            splits.append((train_idx, test_idx))
    
    for train_idx, test_idx in splits:

        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        clsf = LogisticRegression()
        clsf.fit(X_train, y_train)

        y_hat = clsf.predict(X_test)
        probs = clsf.predict_proba(X_test)  # matrix of probabilities assigned to each class

        cfile = CLASSIF_DATA_PATH + '/' + 'logistic.pkl'
        _ = joblib.dump(clsf, cfile, compress=9)

        # load classifier
        cfile = CLASSIF_DATA_PATH + '/' + 'logistic.pkl'
        clsf = joblib.load(cfile)

        score_all.append(clsf.score(X_test, y_test))
        cm_all.append(confusion_matrix(y_test, y_hat))

        prcurve_p = []
        prcurve_r = []
        prcurve_thresh = []
        prcurve_auc = []

        roc_tpr = []
        roc_fpr = []
        roc_thresh = []
        roc_auc = []

        for label in numpy.unique(y):
            y_label_test = numpy.asarray(y_test == label, dtype=int)
            y_label_hat = numpy.asarray(y_hat == label, dtype=int)
            h_label = probs[:, label]

            [prcurve_label_p, prcurve_label_r, prcurve_label_thresh] = precision_recall_curve(y_label_test, h_label)
            [roc_curve_label_fpr, roc_curve_label_tpr, roc_curve_label_thresh] = roc_curve(y_label_test, h_label)

            prcurve_label_p = numpy.insert(prcurve_label_p, 0, 0)
            prcurve_label_r = numpy.insert(prcurve_label_r, 0, 1)
            roc_curve_label_tpr = numpy.insert(roc_curve_label_tpr, 0, 0)
            roc_curve_label_fpr = numpy.insert(roc_curve_label_fpr, 0, 0)

            prcurve_p.append(prcurve_label_p)
            prcurve_r.append(prcurve_label_r)
            prcurve_auc.append(auc(prcurve_label_r, prcurve_label_p))

            roc_tpr.append(roc_curve_label_tpr)
            roc_fpr.append(roc_curve_label_fpr)
            roc_auc.append(auc(roc_curve_label_fpr, roc_curve_label_tpr))

        prcurve_p_all.append(prcurve_p)
        prcurve_r_all.append(prcurve_r)
        prcurve_score_all.append(prcurve_auc)
        roc_tpr_all.append(roc_tpr)
        roc_fpr_all.append(roc_fpr)
        roc_score_all.append(roc_auc)

    numpy.savez(TEST_EVAL_PATH + '/' + 'logistic', score_all=score_all, cm_all=cm_all, prcurve_p_all=prcurve_p_all,
                prcurve_r_all=prcurve_r_all,
                prcurve_score_all=prcurve_score_all, roc_tpr_all=roc_tpr_all, roc_fpr_all=roc_fpr_all,
                roc_score_all=roc_score_all)


# load test results

npzfile = numpy.load(TEST_EVAL_PATH + '/' + 'logistic.npz')

score_all = npzfile['score_all']
cm_all = npzfile['cm_all']
prcurve_p_all = npzfile['prcurve_p_all']
prcurve_r_all = npzfile['prcurve_r_all']
prcurve_score_all = npzfile['prcurve_score_all']
roc_tpr_all = npzfile['roc_tpr_all']
roc_fpr_all = npzfile['roc_fpr_all']
roc_score_all = npzfile['roc_score_all']


# results presentation
if SKIP_GRAPH:
    pass
else:
    # get median of accuracies of all K-folds and output that classifier as result
    fold = numpy.argsort(score_all)[len(score_all) / 2]

    for label in numpy.unique(y):
        c2plot_prc(prcurve_r_all[fold][label], prcurve_p_all[fold][label],
                   'P/R curve (AUC = %s) for %s vs all' % (prcurve_score_all[fold][label], genres[label]),
                   genres[label])
        c2plot_roc(roc_fpr_all[fold][label], roc_tpr_all[fold][label],
                   'ROC curve (AUC = %s) for %s vs all' % (roc_score_all[fold][label], genres[label]),
                   genres[label])

    c2plot_confmat(cm_all[fold].astype(float) / numpy.sum(cm_all[fold], axis=0), genres, 'Normalized confusion matrix')
pass