import cPickle
import numpy as np
import util

def logistic(z):
    """The logistic function, applied elementwise."""
    return 1. / (1. + np.exp(-z))

def softsign(z):
    """The softsign function, applied elementwise."""
    return z / (1. + np.abs(z))

def softsign_prime(z):
    """This function should compute the derivative of the softsign function with
    respect to its argument. It should be applied elementwise to the input array z."""

    ### YOUR CODE HERE
    return 1. / (1. + np.abs(z)) ** 2

def cross_entropy(y, t):
    """The cross-entropy function, applied elementwise."""
    return -t * np.log(y) - (1. - t) * np.log(1. - y)


class LogisticCrossEntropy:
    """The loss function used in logistic regression, as a function of z. That is,

        y = logistic(z)
        loss = cross_entropy(y, t)"""

    @staticmethod
    def value(z, t):
        """The value of the loss, computed elementwise"""
        return np.where(t, np.logaddexp(0., -z), np.logaddexp(0., z))

    @staticmethod
    def derivatives(z, t):
        """The loss derivatives, computed elementwise"""
        y = logistic(z)
        return y - t


class LogisticSquaredError:
    """The value of the squared error loss after a logistic activation function,
    as a function of z. That is,

        y = logistic(z)
        loss = (y - t) ** 2"""

    @staticmethod
    def value(z, t):
        """The value of the loss, computed elementwise."""
        y = logistic(z)
        return 0.5 * (y - t) ** 2

    @staticmethod
    def derivatives(z, t):
        """The loss derivatives, computed elementwise"""
        y = logistic(z)
        dLdy = y - t
        dydz = y * (1. - y)
        return dLdy * dydz
    
class SoftsignCrossEntropy:
    """The value of the cross-entropy loss function after the transformed softsign
    activation function. That is.

        s = softsign(z)
        y = 0.5 * (s + 1.)
        loss = cross_entropy(y, t)"""

    @staticmethod
    def value(z, t):
        """The value of the loss, computed elementwise."""
        y = 0.5 * (1. + softsign(z))
        return cross_entropy(y, t)

    @staticmethod
    def derivatives(z, t):
        """This function should compute the loss derivative for z and t, evaluated elementwise.
        You may wish to refer to LogisticCrossEntropy.derivatives and LogisticSquaredError.derivatives
        for inspiration."""

        ### YOUR CODE HERE
        ###
        ### Hint: you may want to structure it like
        ###
        ###   y = ...
        ###   dLdy = ...
        ###   dyds = ...
        ###   dsdz = ...
        ###   return ...
        
        y = 0.5 * (1. + softsign(z))
        dLdy = (y - t) / ((1. - y) * y)
        dyds = 0.5
        dsdz = softsign_prime(z)
        return dLdy * dyds * dsdz

class LinearModel:
    """The model where z computed as a linear function of x, i.e.

        z = Wx + b."""
    
    def __init__(self, w, b):
        self.w = w
        self.b = b

    # Note: this function is a class method, which means the first argument (cls)
    # is the class itself, i.e. LinearModel.
    @classmethod
    def random_init(cls, num_inputs, std):
        """Initialize the weights and biases to be Gaussian with standard deviation std."""
        w = np.random.normal(0., std, size=num_inputs)
        b = np.random.normal(0., std)
        return cls(w, b)
        
    def compute_activations(self, X):
        """Compute the activations of the output units. Return a dict of the activations."""
        z = np.dot(X, self.w) + self.b
        return {'z': z}

    def get_predictions(self, act):
        """Compute the predictions y given z by thresholding at 0."""
        return act['z'] > 0.

    def cost_derivatives(self, X, act, dLdz):
        """Compute the derivatives of the cost function with respect to the parameters. The
        cost is simply the average loss over all training examples."""
        derivs = {}
        derivs['w'] = np.dot(dLdz, X) / X.shape[0]
        derivs['b'] = dLdz.mean()
        return derivs

    def gradient_descent_update(self, derivs, learning_rate):
        """Given the computed derivatives, apply the gradient descent update rule."""
        self.w -= learning_rate * derivs['w']
        self.b -= learning_rate * derivs['b']

    def apply_weight_decay(self, decay_param, learning_rate):
        self.w *= 1. - decay_param * learning_rate


class MultilayerPerceptron:
    def __init__(self, W1, b1, w2, b2):
        self.W1 = W1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    # Note: this function is a class method, which means the first argument (cls)
    # is the class itself, i.e. LinearModel.
    @classmethod
    def random_init(cls, num_inputs, num_hid, std):
        """Initialize the weights and biases to be Gaussian with standard deviation std."""
        W1 = np.random.normal(0., std, size=(num_hid, num_inputs))
        b1 = np.random.normal(0., std, size=num_hid)
        w2 = np.random.normal(0., std, size=num_hid)
        b2 = np.random.normal(0., std)
        return cls(W1, b1, w2, b2)
        
    def compute_activations(self, X):
        """Compute the activations of the hidden units and output units. Return a dict of the activations."""
        R = np.dot(X, self.W1.T) + self.b1.reshape(1, -1)
        H = softsign(R)
        z = np.dot(H, self.w2) + self.b2
        return {'R': R, 'H': H, 'z': z}

    def get_predictions(self, act):
        """Compute the predictions y given z by thresholding at 0."""
        return act['z'] > 0.

    def cost_derivatives(self, X, act, dLdz):
        """You need to compute the cost derivatives with respect to the parameters of the network. The cost
        is the loss averaged over the training examples. It should output a dict 'derivs' giving the derivatives with
        respect to each of the parameters. E.g., derivs['W1'] should be a matrix, where each entry gives the
        derivative with respect to the corresponding entry of W1; derivs['b2'] should be a scalar giving the
        derivative with respect to b2; and so on.

        We follow the conventions for the sizes of matrices which we've used throughout the course. X is an
        M x D matrix, where M is the size of the mini-batch, and D is the number of input dimensions. In general,
        the first dimension of each of the activation matrices will be the mini-batch size. The weight matrix W1
        is N x D, where N is the number of hidden units. b1 and w2 are both vectors, and b2 is a scalar."""

        derivs = {}

        dLdz = dLdz.reshape(-1, 1) # M * 1
        M = dLdz.shape[0]

        dLdH = dLdz * self.w2.reshape(1, -1) # M x N
        dLdR = dLdH * softsign_prime(act['R']) # M x N
        
        # dLdz * act[H] M x N
        derivs['w2'] = np.sum(dLdz * act['H'], axis=0) / M # N x 1
        
        derivs['b2'] = np.sum(dLdz.reshape((-1, ))) / M

        # np.einsum('xi,xj->xij', A, B) means row-wise outer product
        # derivs['W1'] = np.einsum('xi,xj->xij', dLdR, X).sum(axis = 0) / M # N x D
        derivs['W1'] = np.dot(dLdR.T, X) / M

        derivs['b1'] = np.sum(dLdR, axis=0) / M # N x 1

        return derivs

    def gradient_descent_update(self, derivs, learning_rate):
        """Given the computed derivatives, apply the gradient descent update rule."""
        self.W1 -= learning_rate * derivs['W1']
        self.b1 -= learning_rate * derivs['b1']
        self.w2 -= learning_rate * derivs['w2']
        self.b2 -= learning_rate * derivs['b2']

    def apply_weight_decay(self, decay_param, learning_rate):
        self.W1 *= 1. - decay_param * learning_rate
        self.w2 *= 1. - decay_param * learning_rate


def check_derivatives():
    # check the softsign derivatives
    print 'Checking the softsign derivatives...'
    z = np.random.normal(size=100)
    t = np.random.binomial(1, 0.5, size=100)    
    util.check_loss_derivatives(SoftsignCrossEntropy, z, t)

    # check the backprop computations
    X = np.random.normal(size=(10, 5))
    t = np.random.binomial(1, 0.5, size=10)    
    model = MultilayerPerceptron.random_init(5, 4, 0.1)
    loss_fn = LogisticCrossEntropy
    print

    print 'Checking W1 derivatives...'
    util.check_cost_derivatives(model, loss_fn, 'W1', X, t)
    print
    print 'Checking b1 derivatives...'
    util.check_cost_derivatives(model, loss_fn, 'b1', X, t)
    print
    print 'Checking w2 derivatives...'
    util.check_cost_derivatives(model, loss_fn, 'w2', X, t)
    print
    print 'Checking b2 derivatives...'
    util.check_cost_derivatives(model, loss_fn, 'b2', X, t)
    print

def print_derivatives():
    z = np.array([0.375, -1.223, 2.511, -0.952])
    t = np.array([0, 0, 1, 1])
    print 'dLdz =', SoftsignCrossEntropy.derivatives(z, t)

    data = cPickle.load(open('data.pk', 'rb'))
    model = cPickle.load(open('trained_model.pk', 'rb'))
    X = data['X_train'][:100, :]
    t = data['t_train_noisy'][:100]
    act = model.compute_activations(X)
    dLdz = LogisticCrossEntropy.derivatives(act['z'], t)
    derivs = model.cost_derivatives(X, act, dLdz)

    print 'dEdW1[11, 487] =', derivs['W1'][11, 487]
    print 'dEdW1[74, 434] =', derivs['W1'][74, 434]
    print 'dEdb1[91] =', derivs['b1'][91]
    print 'dEdb1[32] =', derivs['b1'][32]
    print 'dEdw2[55] =', derivs['w2'][55]
    print 'dEdw2[44] =', derivs['w2'][44]
    print 'dEdb2 =', derivs['b2']


def train_model(model, loss_fn, X_train, t_train, X_val, t_val, num_epochs=1000, learning_rate=0.1, batch_size=100,
                weight_decay=0.001, print_every=10):
    for ep in range(num_epochs):
        for X_batch, t_batch in util.get_batches(X_train, t_train, batch_size):
            # compute the activations
            act = model.compute_activations(X_batch)

            # do the gradient descent update
            dLdz = loss_fn.derivatives(act['z'], t_batch)
            param_derivs = model.cost_derivatives(X_batch, act, dLdz)
            model.gradient_descent_update(param_derivs, learning_rate)

            # apply weight decay
            model.apply_weight_decay(weight_decay, learning_rate)

        if ep % print_every == 0:
            # evaluate the training loss and error
            act = model.compute_activations(X_train)
            train_loss = loss_fn.value(act['z'], t_train).mean()
            y = model.get_predictions(act)
            train_err = np.mean(y != t_train)

            # evaluate the validation loss and error
            act = model.compute_activations(X_val)
            val_loss = loss_fn.value(act['z'], t_val).mean()
            y = model.get_predictions(act)
            val_err = np.mean(y != t_val)

            print 'Epoch {}; train_loss={:1.5f}, train_err={:1.5f}, val_loss={:1.5f}, val_err={:1.5f}'.format(
                ep, train_loss, train_err, val_loss, val_err)

    return model

        
        
def train_from_scratch(model_str, loss_str, noisy=False, init_std=0.1, num_hid=100, init_model=None, **kwargs):
    data = cPickle.load(open('data.pk', 'rb'))
    
    if noisy:
        t_train = data['t_train_noisy']
    else:
        t_train = data['t_train_clean']

    loss_fn = {'logistic_ce': LogisticCrossEntropy,
               'logistic_se': LogisticSquaredError,
               'softsign_ce': SoftsignCrossEntropy,
               }[loss_str]

    if init_model is not None:
        model = init_model
    else:
        if model_str == 'linear':
            model = LinearModel.random_init(data['X_train'].shape[1], init_std)
        elif model_str == 'mlp':
            model = MultilayerPerceptron.random_init(data['X_train'].shape[1], num_hid, init_std)
        else:
            raise RuntimeError('Unknown model class: {}'.format(model_str))

    return train_model(model, loss_fn, data['X_train'], t_train, data['X_val'], data['t_val'], **kwargs)
    

# code for training
def run_logisticse():
    train_from_scratch('mlp', 'logistic_se', init_std=10)

def run_logisticxe():
    train_from_scratch('mlp', 'logistic_ce', init_std=10)

def run_softsignxe():
    train_from_scratch('mlp', 'softsign_ce', init_std=10)   
    
    
def run_logisticse_n():
    train_from_scratch('mlp', 'logistic_se', noisy=True)
    
def run_logisticxe_n():
    train_from_scratch('mlp', 'logistic_ce', noisy=True)

def run_softsignxe_n():
    train_from_scratch('mlp', 'softsign_ce', noisy=True)





        

    




