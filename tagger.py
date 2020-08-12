import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    """
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
	###################################################
	# Edit here
    
    state_dict = {}
    for i in range(0,len(tags)):
        state_dict[tags[i]] = i
        
    obs_dict = {}
    unique_words_counter = 0
    
    for sentence in train_data:
        for word in sentence.words:
            if word not in obs_dict:
                obs_dict[word] = unique_words_counter
                unique_words_counter += 1

                
    S = len(state_dict.keys())
    pi = np.zeros([S])
    
    for sentence in train_data:
        pi[state_dict[sentence.tags[0]]] = pi[state_dict[sentence.tags[0]]] + 1
    pi = pi/ np.sum(pi)
    
    A = np.zeros([S,S])
    
    for sentence in train_data:
        for k in range(0, sentence.length - 1):
            i = state_dict[sentence.tags[k]]
            j = state_dict[sentence.tags[k+1]]
            A[i,j] += 1
            
    
    for j in range(0,S):
        s = np.sum(A[j,:])
        if s != 0:
            A[j,:] = A[j,:]/s
    
    B = np.zeros([S, len(obs_dict.keys())])
    
    for sentence in train_data:
        for j in range(0, sentence.length):
            row = state_dict[sentence.tags[j]]
            col = obs_dict[sentence.words[j]]
            B[row,col] += 1
            
    
    for j in range(0,S):
        su = np.sum(B[j,:])
        if su != 0:
            B[j,:] = B[j,:]/ su  
    
    model = HMM(pi, A, B, obs_dict, state_dict)
     
	###################################################
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
	###################################################
	# Edit here
    col = np.ones([len(model.state_dict.keys()),1])*0.000001
    
    for sentence in test_data:
        for word in sentence.words:
            if word not in model.obs_dict:
                model.B = np.append(model.B, col,axis=1)
                model.obs_dict[word] = len(model.obs_dict.keys())
                         
    for sentence in test_data:           
        t = model.viterbi(sentence.words)
        tagging.append(t)
	###################################################
    return tagging