from model.model_basic import BasicStaticModel
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from skift import FirstColFtClassifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class SVCClassifier(BasicStaticModel):

    def __init__(self, name='basicModel', n_folds=5, config=None):
        BasicStaticModel.__init__(self, name=name, n_folds=n_folds, config=config)

    def create_model(self):
        classifier = SVC(kernel="rbf")
        classifier = CalibratedClassifierCV(classifier)
        classifier = SVC(kernel="linear")
        self.classifier = classifier
        self.classifier = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)
        return self.classifier


class Fasttext(BasicStaticModel):
    def __init__(self, name='basicModel', n_folds=5, config=None):
        BasicStaticModel.__init__(self, name=name, n_folds=n_folds, config=config)

    def create_model(self):
        sk_clf = FirstColFtClassifier(lr=1.0, epoch=10,
                                      wordNgrams=1,
                                      minCount=5, verbose=2)
        return sk_clf



