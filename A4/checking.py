import numpy as np

import mixture
import util



def multinomial_entropy(p):
    """Compute the entropy of a Bernoulli random variable, in nats rather than bits."""
    p = np.clip(p, 1e-20, np.infty)      # avoid taking the log of 0
    return -np.sum(p * np.log(p))


def variational_objective(model, X, R, pi, theta):
    """Compute the variational lower bound on the log-likelihood that each step of E-M
    is maximizing. This is described in the paper

        Neal and Hinton, 1998. A view of the E-M algorithm that justifies incremental, sparse, and other variants.

    We can test the update rules by verifying that each step maximizes this bound.
    """

    model = mixture.Model(model.prior, mixture.Params(pi, theta))
    expected_log_prob = model.expected_joint_log_probability(X, R)
    entropy_term = np.sum(multinomial_entropy(R))
    return expected_log_prob + entropy_term

def perturb_pi(pi, eps=1e-6):
    pi = np.random.normal(pi, eps)
    pi = np.clip(pi, 1e-10, np.infty)
    pi /= pi.sum()
    return pi

def perturb_theta(theta, eps=1e-6):
    theta = np.random.normal(theta, eps)
    theta = np.clip(theta, 1e-10, 1. - 1e-10)
    return theta

def perturb_R(R, eps=1e-6):
    R = np.random.normal(R, eps)
    R = np.clip(R, 1e-10, np.infty)
    R /= R.sum(1).reshape((-1, 1))
    return R



def check_m_step():
    """Check that the M-step updates by making sure they maximize the variational
    objective with respect to the model parameters."""
    np.random.seed(0)

    NUM_IMAGES = 100

    X = util.read_mnist_images(mixture.TRAIN_IMAGES_FILE)
    X = X[:NUM_IMAGES, :]
    R = np.random.uniform(size=(NUM_IMAGES, 10))
    R /= R.sum(1).reshape((-1, 1))
    model = mixture.Model.random_initialization(mixture.Prior.default_prior(), 10, 784)

    theta = model.update_theta(X, R)
    pi = model.update_pi(R)

    opt = variational_objective(model, X, R, pi, theta)
    
    ok = True
    for i in range(20):
        new_theta = perturb_theta(theta)
        new_obj = variational_objective(model, X, R, pi, new_theta)
        if new_obj > opt:
            ok = False
    if ok:
        print 'The theta update seems OK.'
    else:
        print 'Something seems to be wrong with the theta update.'

    if not np.allclose(np.sum(pi), 1.):
        print 'Uh-oh. pi does not seem to sum to 1.'
    else:
        ok = True
        for i in range(20):
            new_pi = perturb_pi(pi)
            new_obj = variational_objective(model, X, R, new_pi, theta)
            if new_obj > opt:
                ok = False
        if ok:
            print 'The pi update seems OK.'
        else:
            print 'Something seems to be wrong with the pi update.'


def check_e_step():
    """Check the E-step updates by making sure they maximize the variational
    objective with respect to the responsibilities. Note that this does not
    fully check your solution to Part 2, since it only applies to fully observed
    images."""
    
    np.random.seed(0)

    NUM_IMAGES = 100

    X = util.read_mnist_images(mixture.TRAIN_IMAGES_FILE)
    X = X[:NUM_IMAGES, :]
    model = mixture.train_from_labels(show=False)

    # reduce the number of observations so that the posterior is less peaked
    X = X[:, ::50]
    model.params.theta = model.params.theta[:, ::50]
    
    R = model.compute_posterior(X)

    opt = variational_objective(model, X, R, model.params.pi, model.params.theta)

    if not np.allclose(R.sum(1), 1.):
        print 'Uh-oh. Rows of R do not seem to sum to 1.'
    else:
        ok = True
        for i in range(20):
            new_R = perturb_R(R)
            new_obj = variational_objective(model, X, new_R, model.params.pi, model.params.theta)
            if new_obj > opt:
                ok = False
        if ok:
            print 'The E-step seems OK.'
        else:
            print 'Something seems to be wrong with the E-step.'




