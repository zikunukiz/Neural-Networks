# Additive multimodal log-bilinear model

import sys
import time
import lm_tools
import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.sparse import vstack
from autograd import grad

class MLBL(object):
    """
    Multimodal Log-bilinear language model trained using SGD
    """
    def __init__(self,
                 name='lbl',
                 loc='models/mlbl.pkl',
                 seed=1234,                 
                 k=5,
                 V=10364,
                 K=50,
                 D=10,
                 h=3,
                 context=5,
                 batchsize=20,
                 maxepoch=10,
                 eta_t=0.2,
                 gamma_r=0.0,
                 gamma_c=0.0,
                 f=0.995,
                 p_i=0.5,
                 p_f=0.5,
                 T=20,
                 verbose=1):
        """
        name: name of the network
        loc: location to save model files
        seed: random seed        
        k: validation interval before stopping
        V: vocabulary size
        K: embedding dimensionality
        D: dimensionality of the image features
        h: intermediate layer dimensionality
        context: word context length
        batchsize: size of the minibatches
        maxepoch: max number of training epochs
        eta_t: learning rate
        gamma_r: weight decay for representations
        gamma_c: weight decay for contexts
        f: learning rate decay
        p_i: initial momentum
        p_f: final momentum
        T: number of epochs until p_f is reached (linearly)
        verbose: display progress
        """
        self.name = name
        self.loc = loc        
        self.seed = seed
        self.k = k
        self.V = V
        self.K = K
        self.D = D
        self.h = h
        self.context = context
        self.batchsize = batchsize
        self.maxepoch = maxepoch
        self.eta_t = eta_t
        self.gamma_r = gamma_r
        self.gamma_c = gamma_c
        self.f = f
        self.p_i = p_i
        self.p_f = p_f
        self.T = T
        self.verbose = verbose
        self.p_t = (1 - (1 / T)) * p_i + (1 / T) * p_f 


    def init_params(self, embed_map, count_dict):
        """
        Initializes embeddings and context matricies
        """        
        rs = npr.RandomState(self.seed)

        # Pre-trained word embedding matrix
        if embed_map != None:
            R = np.zeros((self.K, self.V))
            for i in range(self.V):
                word = count_dict[i]
                if word in embed_map:
                    R[:,i] = embed_map[word]
                else:
                    R[:,i] = embed_map['*UNKNOWN*']
        else:
            r = np.sqrt(6) / np.sqrt(self.K + self.V + 1)
            R = rs.rand(self.K, self.V) * 2 * r - r
        bw = np.zeros((1, self.V))

        self.params = []

        # Context 
        C = 0.01 * rs.randn(self.context, self.K, self.K)

        # Image context
        M = 0.01 * rs.randn(self.h, self.K)

        # Hidden layer
        r = np.sqrt(6) / np.sqrt(self.D + self.h + 1)
        J = rs.rand(self.D, self.h) * 2 * r - r
        bj = np.zeros((1, self.h))

        # Initial deltas used for SGD
        self.deltaR = np.zeros(np.shape(R))
        self.deltaC = np.zeros(np.shape(C))
        self.deltaB = np.zeros(np.shape(bw))
        self.deltaM = np.zeros(np.shape(M))
        self.deltaJ = np.zeros(np.shape(J))
        self.deltaBj = np.zeros(np.shape(bj))

        # Pack up
        self.params += [R]
        self.params += [C]
        self.params += [bw]
        self.params += [M]
        self.params += [J]
        self.params += [bj]

    def forward(self, params, X, Im, test=True):
        """
        Feed-forward pass through the model

        Args:
            params: list of weight matrix
            X: word indices, 
            Im: images, 
            test: flag, whether test or train
        """
        batchsize = X.shape[0]

        R = params[0]   # word representations, variable 'R' in the paper
        C = params[1]   # context parameter of text modality, variable 'C^i' in the paper
        bw = params[2]  # bias of text modality, variable 'b' in the paper
        M = params[3]   # context parameter of image modality, 'C^m' in the paper
        J = params[4]   # weight of image modality, 'J' in the paper
        bj = params[5]  # bias of image modality, 'h' in the paper

        ########################################################################
        # You should implement forward pass here!        
        # preds: softmax output 
        ########################################################################
        
        # preds = ...

        contextsize = X.shape[1]
        context_indices = np.arange(contextsize)
        batch_indices = np.arange(batchsize)
        
        rw= R[:,X[batch_indices].T].T.reshape(batchsize, contextsize, 1, -1 )
            
        Crw = np.concatenate(np.sum(np.dot(rw, C)[j, context_indices, :,context_indices, :], axis=0) 
            for j in range(0, batchsize))
        
        ReLU = np.maximum(np.dot(Im, J) + bj, 0) # batch size x embedding dimensionality
        
        rhat = Crw + np.dot(ReLU, M) # batchsize x embedding dimensionality

        rhatTri = np.dot(rhat, R) + bw # vocab size x batchsize
        
        denominator = np.sum(np.exp(rhatTri), axis=1).reshape(-1, 1) # batchsize x 1
        
        preds = np.exp(rhatTri)/denominator # batchsize x vocabulary size

        ########################################################################

        return preds 

    def objective(self, Y, preds):
        """
        Compute the objective function
        """
        batchsize = Y.shape[0]

        # Cross-entropy
        logits = np.log(preds + 1.0e-20)
        C = -np.sum(Y * logits) / batchsize

        return C

    def compute_obj(self, params, X, Im, Y):
        """
        Perform a forward pass and compute the objective
        """
        preds = self.forward(params, X, Im, test=False)
        obj = self.objective(Y, preds)
        return obj

    def backward(self, params, X, Im, Y):
        """
        Backward pass through the network
        """
        ########################################################################
        # You should implement backward pass here!
        # grad_params: list of the gradient
        # grad_params[0]: gradient of R
        # grad_params[1]: gradient of C
        # grad_params[2]: gradient of bw
        # grad_params[3]: gradient of M
        # grad_params[4]: gradient of J
        # grad_params[5]: gradient of bj
        ########################################################################

        # grad_params = ...      
        
        grad_fun = grad(self.compute_obj)
        grad_params = grad_fun(params, X, Im, Y)
        
        ########################################################################
        
        dR = grad_params[0]
        dC = grad_params[1]
        db = grad_params[2]
        dM = grad_params[3]
        dJ = grad_params[4]
        dBj = grad_params[5]

        # Weight decay terms
        dR += self.gamma_r * params[0]
        dC += self.gamma_c * params[1]
        dM += self.gamma_c * params[3]
        dJ += self.gamma_c * params[4]

        # Pack
        self.dR = dR
        self.dM = dM
        self.db = db
        self.dC = dC
        self.dJ = dJ
        self.dBj = dBj

        return grad_params

    def update_params(self, X):
        """
        Update the network parameters using the computed gradients
        """
        batchsize = X.shape[0]
        self.deltaC = self.p_t * self.deltaC - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dC
        self.deltaR = self.p_t * self.deltaR - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dR
        self.deltaB = self.p_t * self.deltaB - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.db
        self.deltaM = self.p_t * self.deltaM - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dM
        self.deltaJ = self.p_t * self.deltaJ - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dJ
        self.deltaBj = self.p_t * self.deltaBj - \
            (1 - self.p_t) * (self.eta_t / batchsize) * self.dBj


        self.params[0] += self.deltaR
        self.params[1] += self.deltaC
        self.params[2] += self.deltaB
        self.params[3] += self.deltaM
        self.params[4] += self.deltaJ
        self.params[5] += self.deltaBj

    def update_hyperparams(self):
        """
        Updates the learning rate and momentum schedules
        """
        self.eta_t *= self.f
        if self.step < self.T:
            self.p_t = (1 - ((self.step + 1) / self.T)) * self.p_i + \
                ((self.step + 1) / self.T) * self.p_f
        else:
            self.p_t = self.p_f

    def train(self, X, indX, XY, V, indV, VY, IM, VIM, count_dict, word_dict, embed_map, prog):
        """
        Trains the LBL
        """
        ###
        global B, x
        B = np.array([])
        x = np.array([])
        
        self.start = self.seed
        self.init_params(embed_map, count_dict)
        self.step = 0
        inds = np.arange(len(X))
        numbatches = len(inds) / self.batchsize
        tic = time.time()
        bleu = [0.0]*4
        best = 0.0
        scores = '/'.join([str(b) for b in bleu])
        patience = 10
        count = 0
        done = False

        # Main loop
        lm_tools.display_phase(1)
        for epoch in range(self.maxepoch):
            if done:
                break
            self.epoch = epoch            
            prng = npr.RandomState(self.seed + epoch + 1)
            prng.shuffle(inds)
            for minibatch in range(numbatches):

                batchX = X[inds[minibatch::numbatches]]
                batchY = XY[inds[minibatch::numbatches]].toarray()
                batchindX = indX[inds[minibatch::numbatches]].astype(int).flatten()
                batchindX = np.floor(batchindX/5).astype(int)
                batchIm = IM[batchindX]

                loss_val = self.compute_obj(self.params, batchX, batchIm, batchY)                
                self.backward(self.params, batchX, batchIm, batchY)
                self.update_params(batchX)
                
                if np.isnan(loss_val):
                    print 'NaNs... breaking out'
                    done = True
                    break

                # Print out progress
                if np.mod(minibatch * self.batchsize, prog['_details']) == 0 and minibatch > 0:
                    print "epoch/pts: %04d/%05d, cross-entropy loss: %.2f, time: %.2f" % (epoch+1, minibatch * self.batchsize, loss_val, (time.time()-tic)/60)
                if np.mod(minibatch * self.batchsize, prog['_samples']) == 0 and minibatch > 0:
                    print "best: %s" % (scores)
                    print '\nSamples:'
                    # lm_tools.generate_and_show(self, word_dict, count_dict, VIM, k=3)
                    VIM_example = VIM[prog['_val_example_idx'], :]
                    VIM_file_list = prog['_val_example_file']
                    lm_tools.generate_and_show_html(self, word_dict, count_dict, VIM_example, VIM_file_list, show=prog['_show_browser'], k=3)
                    print ' '
                if np.mod(minibatch * self.batchsize, prog['_update']) == 0 and minibatch > 0:
                    self.update_hyperparams()
                    self.step += 1
                    print "learning rate: %.4f, momentum: %.4f" % (self.eta_t, self.p_t)

                # Compute BLEU
                if np.mod(minibatch * self.batchsize, prog['_bleu']) == 0 and minibatch > 0:
                    bleu = lm_tools.compute_bleu(self, word_dict, count_dict, VIM, prog, k=3)
                    
                    ###
                    B = np.append(B, bleu[-1])
                    x = np.append(x, minibatch * self.batchsize)
                    
                    if bleu[-1] >= best:
                        count = 0
                        best = bleu[-1]
                        scores = '/'.join([str(b) for b in bleu])
                        print 'bleu score = {}'.format(bleu[-1])
                        lm_tools.save_model(self, self.loc)
                        
                    else:
                        count += 1
                        if count == patience:
                            done = True
                            break

            self.update_hyperparams()
            self.step += 1
        return best


def test_MLBL_implementation():
    """ Test Your Implementation of Forward and Backward """

    import coco_proc
    import trainer
    import cPickle as pickle

    z, zd, zt = coco_proc.process(context=5)

    d = {}    
    d['name'] = 'testrun'    
    d['loc'] = 'models/mlbl_model.pkl'
    d['context'] = 5
    d['learning_rate'] = 0.43
    d['momentum'] = 0.23
    d['batch_size'] = 40
    d['maxepoch'] = 10
    d['hidden_size'] = 441
    d['word_decay'] = 3e-7
    d['context_decay'] = 1e-8
    d['factors'] = 50
    
    # Load the word embeddings
    embed_map = trainer.load_embeddings()

    # Unpack some stuff from the data
    train_ngrams = z['ngrams']
    train_labels = z['labels']
    train_instances = z['instances']
    word_dict = z['word_dict']
    index_dict = z['index_dict']
    context = z['context']
    vocabsize = len(z['word_dict'])
    trainIM = z['IM']
    train_index = z['index']
    
    net = MLBL(name=d['name'],
                    loc=d['loc'],
                    seed=1234,                    
                    V=vocabsize,
                    K=100,
                    D=trainIM.shape[1],
                    h=d['hidden_size'],
                    context=d['context'],
                    batchsize=1,
                    maxepoch=d['maxepoch'],
                    eta_t=d['learning_rate'],
                    gamma_r=d['word_decay'],
                    gamma_c=d['context_decay'],
                    f=0.99,
                    p_i=d['momentum'],
                    p_f=d['momentum'],
                    T=20.0,
                    verbose=1
                    )

    # Train the network
    X = train_instances
    indX = train_index
    Y = train_labels

    net.init_params(embed_map, index_dict)

    context_size = d['context']
    batchX = X[0:context_size]
    batchY = Y[0:context_size].toarray()
    batchindX = indX[0:context_size].astype(int).flatten()
    batchindX = np.floor(batchindX/5).astype(int)
    batchIm = trainIM[batchindX]

    # check forward implementation
    ft = net.forward(net.params, batchX, batchIm, test=True)
    
    # load gt feature
    ft_gt = pickle.load(open("data/val_implementation.p", "rb"))
    
    # it should be less than 1.0e-5
    print 'Difference (L2 norm) between implemented and ground truth feature = {}'.format(np.linalg.norm(ft_gt - ft))

if __name__ == '__main__':

    test_MLBL_implementation()
