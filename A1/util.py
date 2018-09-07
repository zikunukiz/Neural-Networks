import copy
import numpy as np
import pylab


def get_batches(X, t, batch_size):
    ndata = X.shape[0]
    assert ndata % batch_size == 0
    assert t.shape == (ndata,)

    start = 0
    while start + batch_size <= ndata:
        end = start + batch_size
        yield X[start:end, :], t[start:end]
        start += batch_size


def check_loss_derivatives(loss_fn, z, t, eps=1e-8):
    try:
        exact = loss_fn.derivatives(z, t)
        numerical = (loss_fn.value(z + eps, t) - loss_fn.value(z - eps, t)) / (2. * eps)
        rel_error = np.abs(exact - numerical) / (np.abs(exact) + np.abs(numerical))

        if np.max(rel_error) < 1e-6:
            print 'OK'
        else:
            print 'Relative error of {}: too large'.format(np.max(rel_error))
    except NotImplementedError:
        print 'This part is not done yet.'


def check_cost_derivative(model, loss_fn, param, idx, X, t, eps=1e-8):
    # perturb the chosen parameter by eps
    model_plus = copy.deepcopy(model)
    model_minus = copy.deepcopy(model)
    if idx is not None:
        getattr(model_plus, param)[idx] += eps
        getattr(model_minus, param)[idx] -= eps
    else:
        setattr(model_plus, param, getattr(model_plus, param) + eps)
        setattr(model_minus, param, getattr(model_minus, param) - eps)

    act = model.compute_activations(X)
    dLdz = loss_fn.derivatives(act['z'], t)
    derivs = model.cost_derivatives(X, act, dLdz)
    if idx is not None:
        exact = derivs[param][idx]
    else:
        exact = derivs[param]

    act_plus = model_plus.compute_activations(X)
    loss_plus = loss_fn.value(act_plus['z'], t).mean()
    act_minus = model_minus.compute_activations(X)
    loss_minus = loss_fn.value(act_minus['z'], t).mean()
    numerical = (loss_plus - loss_minus) / (2. * eps)

    rel_error = np.abs(exact - numerical) / (np.abs(exact) + np.abs(numerical) + 0.0001)
    return rel_error

    

def check_cost_derivatives(model, loss_fn, param, X, t, eps=1e-8):
    try:
        val = getattr(model, param)
        if np.isscalar(val):
            idxs = [None]
        elif val.ndim == 1:
            idxs = range(val.shape[0])
        elif val.ndim == 2:
            idxs = [(i, j) for i in range(val.shape[0]) for j in range(val.shape[1])]

        max_rel_error = 0.
        for idx in idxs:
            rel_error = check_cost_derivative(model, loss_fn, param, idx, X, t, eps)
            max_rel_error = max(rel_error, max_rel_error)
        
        if max_rel_error < 1e-4:
            print 'OK'
        else:
            print 'Relative error of {}: too large'.format(max_rel_error)
    except NotImplementedError:
        print 'This part is not done yet.'


def show_images(X, im_shape=(28,28)):
    num = X.shape[0]
    nrows = int(np.sqrt(num))
    ncols = int(np.ceil(num / nrows))

    pylab.figure()
    for i in range(num):
        pylab.subplot(nrows, ncols, i+1)
        pylab.imshow((1. - X[i, :]).reshape(im_shape), cmap='gray')
        pylab.axis('off')


        
    
