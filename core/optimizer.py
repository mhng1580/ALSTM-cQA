from collections import OrderedDict
import theano
import theano.tensor as T

from rnn_enc.utils.tools import np_floatX, _p


class SGD(object):
    """ 
    Stochastic Gradient Descent
    """
    def __init__(self, inputList, params, cost, lr=None):
        updates = [(param, param - lr * grad) for param, grad in
                   zip(params.values(), T.grad(cost, wrt=params.values()))]
        self.f_train = theano.function([lr]+inputList, cost, updates=updates, name='sgd_update')
        self.params = OrderedDict()

    def trainModel(self):
        return self.f_train

    def get_params(self):
        return self.params
    
    @staticmethod
    def get_optType():
        return 'SGD'

class AdaDelta(object):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize
    """
    def __init__(self, inputList, params, cost, lr=None, decay=0.95, eps=1e-6):
        grads = T.grad(cost, wrt=params.values())

        gradParams = [theano.shared(p.get_value() * np_floatX(0.), name=_p(k, 'grad')) for k, p in params.iteritems()]
        rgrad2Params = [theano.shared(p.get_value() * np_floatX(0.), name=_p(k, 'rgrad2')) for k, p in params.iteritems()]
        rup2Params = [theano.shared(p.get_value() * np_floatX(0.), name=_p(k, 'rup2')) for k, p in params.iteritems()]

        gradUpdates = [(g, g_val) for g, g_val in zip(gradParams, grads)]
        rgrad2Updates = [(rg2, decay * rg2 + (1 - decay) * (g_val ** 2)) for rg2, g_val in zip(rgrad2Params, grads)]

        upList = [-T.sqrt(ru2 + eps) / T.sqrt(rg2 + eps) * g
                 for g, ru2, rg2 in zip(gradParams, rup2Params, rgrad2Params)]
        rup2Updates = [(ru2, decay * ru2 + (1-decay) * (up ** 2))
                 for ru2, up in zip(rup2Params, upList)]
        paramUpdates = [(p, p + up) for p, up in zip(params.values(), upList)]

        f_grad = theano.function(inputList, cost, updates=gradUpdates+rgrad2Updates, name='adadelta_f_grad')
        f_update = theano.function([lr], [], updates=rup2Updates+paramUpdates, on_unused_input='ignore', name='adadelta_f_update')
        
        def f_train(lrVal, *inputList):
            cost = f_grad(*inputList)
            f_update(lrVal)
            return cost

        self.f_train = f_train
        self.params = OrderedDict()
        for params in [gradParams, rgrad2Params, rup2Params]:
            for param in params:
                self.params[param.name] = param

    def trainModel(self):
        return self.f_train

    def get_params(self):
        return self.params

    @staticmethod
    def get_optType():
        return 'AdaDelta'

class AdaGrad(object):
    """
    Adaptive Gradient Descent
    """
    def __init__(self, inputList, params, cost, lr=None, eps=1e-6):
        grads = T.grad(cost, wrt=params.values())
        gradParams = [theano.shared(p.get_value() * np_floatX(0.), name=_p(k, 'grad')) for k, p in params.iteritems()]
        agrad2Params = [theano.shared(p.get_value() * np_floatX(0.), name=_p(k, 'agrad2')) for k, p in params.iteritems()]
        gradUpdates = [(g, g_val) for g, g_val in zip(gradParams, grads)]
        agrad2Updates = [(ag2, ag2 + (g_val ** 2)) for ag2, g_val in zip(agrad2Params, grads)]

        paramUpdates = [(p, p - (lr / T.sqrt(ag2 + eps)) * g) for p, g, ag2 in zip(params.values(), gradParams, agrad2Params)]
        
        f_grad = theano.function(inputList, cost, updates=gradUpdates+agrad2Updates, name='adagrad_f_grad')
        f_update = theano.function([lr], [], updates=paramUpdates, name='adagrad_f_update')

        def f_train(lrVal, *inputList):
            cost = f_grad(*inputList)
            f_update(lrVal)
            return cost

        self.f_train = f_train
        self.params = OrderedDict()
        for params in [gradParams, agrad2Params]:
            for param in params:
                self.params[param.name] = params

    def trainModel(self):
        return self.f_train

    def get_params(self):
        return self.params

    @staticmethod
    def get_optType():
        return 'AdaGrad'

