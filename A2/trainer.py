# Trainer module

import os
import config
import cPickle as pickle

from mlbl import MLBL

def trainer(z, zd):
    """
    Trainer function for multimodal log-bilinear models

    Dictionary:
    'name' (name of the model, unique to each run)
    'loc' (location to save)
    'context' int:[3,25]
    'learning_rate' float:[0.001, 10]
    'momentum' float:[0, 0.9]
    'batch_size' int:[20, 100]
    'hidden_size' int:[100, 2000]    
    'word_decay' float:[1e-3, 1e-9]
    'context_decay' float:[1e-3, 1e-9]
    'factors' (mul model only!) int:[50,200], truncate by embedding_size
    """
    d = {}
    d['name'] = 'testrun'    
    d['loc'] = 'models/mlbl_model.pkl'
    d['context'] = 5
    d['learning_rate'] = 0.5
    d['momentum'] = 0.05
    d['batch_size'] = 40
    d['maxepoch'] = 10
    d['hidden_size'] = 441
    d['word_decay'] = 3e-7
    d['context_decay'] = 1e-8
    d['factors'] = 50

    if not os.path.isdir('models'):
        os.mkdir('models')

    if not os.path.isdir('html'):
        os.mkdir('html')

    # Progress display and update times
    # it is measured by the number of iteration in terms of pts
    prog = {}
    prog['_details'] = 100   # How often to display training details
    prog['_samples'] = 2000  # How often to display samples
    prog['_update'] = 50000  # How often to update learning rate schedule
    prog['_bleu'] = 2500     # How often to compute BLEU
    prog['_neval'] = 250      # How many development images to evaluate
    prog['_evaldev'] = True   # Use development set reference captions for eval
    prog['_show_browser'] = True # Show example validation in browser
    prog['_val_example_idx'] = zd['val_example_idx']
    prog['_val_example_file'] = zd['val_example_file']
    
    # Load the word embeddings
    embed_map = load_embeddings()

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
    
    dev_ngrams = zd['ngrams']
    dev_labels = zd['labels']
    dev_instances = zd['instances']
    devIM = zd['IM']
    dev_index = zd['index']
    
    # Initialize the network    
    net = MLBL(name=d['name'],
                    loc=d['loc'],
                    seed=1234,                    
                    V=vocabsize,
                    K=100,
                    D=trainIM.shape[1],
                    h=d['hidden_size'],
                    context=d['context'],
                    batchsize=d['batch_size'],
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
    V = dev_instances
    indV = dev_index
    VY = dev_labels

    best = net.train(X, indX, Y, V, indV, VY, trainIM, devIM, index_dict, word_dict, embed_map, prog)
    return best


def load_embeddings():
    """
    Load in the embeddings
    """
    return pickle.load(open(config.paths['embedding'], 'rb'))
