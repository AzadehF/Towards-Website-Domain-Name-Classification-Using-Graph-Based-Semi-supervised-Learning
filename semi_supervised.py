from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import sparse

class Base(): # Graph-Based Semi-Supervised Learning Implementation from https://github.com/junliangma/gbssl
    __metaclass__ = ABCMeta
    def __init__(self,graph,max_iter=30):
        self.max_iter = max_iter
        self.graph = graph

    @abstractmethod
    def _build_propagation_matrix(self):
        raise NotImplementedError("Propagation matrix construction must be implemented to fit a model.")

    @abstractmethod
    def _build_base_matrix(self):
        raise NotImplementedError("Base matrix construction must be implemented to fit a model.")

    def _init_label_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        return np.zeros((n_samples,n_classes))

    def _arrange_params(self):
        """Do nothing by default"""
        pass

    def fit(self,x,y):
        """Fit a graph-based semi-supervised learning model
        All the input data is provided array X (labeled samples only)
        and corresponding label array y.
        Parameters
        ----------
        x : array_like, shape = [n_labeled_samples]
            Node IDs of labeled samples
        y : array_like, shape = [n_labeled_samples]
            Label IDs of labeled samples
        Returns
        -------
        self : returns an instance of self.
        """
        self.x_ = x
        self.y_ = y

        self._arrange_params()

        self.F_ = self._init_label_matrix()

        self.P_ = self._build_propagation_matrix()
        self.B_ = self._build_base_matrix()

        remaining_iter = self.max_iter
        while remaining_iter > 0:
            self.F_ = self._propagate()
            remaining_iter -= 1

        return self

    def _propagate(self):
        return self.P_.dot(self.F_) + self.B_

    def predict(self,x):
        """Performs prediction based on the fitted model
        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs
        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input node IDs
        """
        probas = self.predict_proba(x)
        return np.argmax(probas,axis=1)

    def predict_proba(self,x):
        """Predict probability for each possible label
        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs
        Returns
        -------
        probabilities : array_like, shape = [n_samples, n_classes]
            Probability distributions across class labels
        """
        return (self.F_[x].T / np.sum(self.F_[x], axis=1)).T


class LGC(Base):
    """Local and Global Consistency (LGC) for GBSSL
    Parameters
    ----------
    alpha : float
      clamping factor
    max_iter : float
      maximum number of iterations allowed
    Attributes
    ----------
    x_ : array, shape = [n_samples]
        input array of node IDs.
    Examples
    --------
    <<<
    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """

    def __init__(self,graph,alpha=0.0001,max_iter=30):
        super(LGC, self).__init__(graph,max_iter=30)
        self.alpha=alpha

    def _build_propagation_matrix(self):
        """ LGC computes the normalized Laplacian as its propagation matrix"""
        D2 = np.sqrt(sparse.diags((1.0/(self.graph.sum(1))).T.tolist()[0],offsets=0))
        S = D2.dot(self.graph).dot(D2)
        return self.alpha*S

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        B = np.zeros((n_samples,n_classes))
        B[self.x_,self.y_] = 1
        return (1-self.alpha)*B

