###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Sathish Nandakumar - satnanda
# Khaled Abou Samak - khabous
# Rohan Kayan - rkkayan
# (Based on skeleton code by D. Crandall)
#
#NOTE: WE HAD TO MAKE CHANGES IN "label.py" and "pos_solver.py", PLEASE BOTH OF THESE PROGRAMS FROM OUR REPOSITORY
# TO TEST/VALIDATE THE CODE
####
# Section 1:
# P(Si) is the initial probability of each part of speech. This is calculated with the below formula
# P(S(i)) = Count of sentences with each POS(Si) at 1st position/Sum of all sentences.
#
# P(S(i+1)|S(i))is the transition probability to transition from part of speech at state i to state i+1. This is
# calculated as given below
# P(S(i+1)|S(i)) = Count of sentences with POS(Si) followed by POS(Si+1) / Sum of all POS(Si)
#
# P(W(i)|S(i)) is the emission probability of a word (observed) given an unobserved POS S. This is calculated as
# give below
# P(W(i)|S(i)) = Count of a word Wi tagged as POS Si / Sum of all POS(Si)
#
# Section 2:
# si = arg max(si)P(Si = si|W) - We have calculated the P(Si = si|Wi) = P(Wi|Si)P(Si) / P(Wi) for each word Wi in all the
# possible POS Si and picked the POS with maximum probability. We have used default value 1e-10 for smoothing as we
# had some unseen emission probabilities in training dataset
#
# Section 3:
# HMM/Viterbi: si...sN = arg max(si...sN)P(Si = si|W) - We used, initial, emission and transition probabilities
# calculated in section 1 to apply it on HMM to calculate the probability for all POS at each state and then backtraced
# to find the most probable sequence using the below derivation
# P(Si) = max{P(Si-1) * P(Si+1|Si) * P(Wi|Si)}
#
# Section 4:
# MCMC sampling for Complex BayesNet - In this BayesNet each state is dependent on previous 2 states so we had
# calculated the posterior of each state Si based on its dependency with other states. We sampled MCMC by sampling
# each state with all possible POS like P(S1=N| S2=V...Sn=A), P(S1=V| S2=V...Sn=A) etc.,
# Sampling POS S at state i (initial position) - P(W1|S1) * P(S1)
# POS S at state i+1 (2nd position) - P(W2|S2) * P(S2|S1=N)
# POS S at state 1+2...i+n-1 (all positions from 3rd can be calculated as ) - P(W3|S3) * P(S3|S2) * P(S3|S1)
#
# Assumptions:
# (a) There were some words, transitions POS prob and emission prob not in training dataset so we had smoothed the
# probablities with a default value of 1e-10
# (b) This program runs for close to 10 mins for all 3 algorithms, the 95% of the time is spent in the MCMC algorithm
# as we sample 1000 records for each sentece in the test dataset so it has to do computations 1000 times more than the
# other 2 algorithms. We believe if we reduce the sample size or reduce the test dataset size the program will execute
# in lesser time
# (c) We made changes in "label.py" to retain individual objects for each algorightm so when we calculate the score
# we can take the right final POS tags and data structures corresponds to each algorithm.
#
# Accuracy
#                   Words correct:     Sentences correct:
#0. Ground truth:      100.00%              100.00%
#      1. Simple:       93.92%               47.45%
#         2. HMM:       95.09%               54.50%
#     3. Complex:       92.65%               41.75%
#
# Best Algorithm:
# HMM/Viterbi has given the best accuracy among the 3 algorithms, Complex algorithms performed better than Simple but
# we belive it has several unseen tranistion and emission probability so the accuracy is lesser than HMM/Viterbi
####

import random
import math
import collections
import sys
import operator
import pandas as pd

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    ip = collections.defaultdict(lambda:0) #Initial probability
    tp = collections.defaultdict(lambda: 0)  # Transition probability
    ep = collections.defaultdict(lambda: 0)  # Emission probability
    poscnt = collections.defaultdict(lambda: 0)  # count of each pos occurrence
    hmmvt = collections.defaultdict(lambda: 0)  # Hmm Viterbit probability
    wcnt = collections.defaultdict(lambda: 0)
    psprob = collections.defaultdict(lambda: 0)
    totalwordcount = 0
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self,model, sentence, label):
        result = 0
        if model=="Simple":
            for x in range(0, len(sentence)):
                try:
                    emission_value = self.ep[(sentence[x], label[x])]
                except:
                    emission_value = 1e-10
                wp = float(self.wcnt[sentence[x]]) / self.totalwordcount
                if wp == 0.0:
                    wp = 1e-10
                denom = math.log(wp)
                nominator = math.log(emission_value * self.psprob[label[x]])
                result += nominator - denom
        elif model == "HMM":
            '''
            State 1:
            P(W1|S1) P(S1)
            
            State 2:
            P(W2|S2) P(S2|S1)
            
            State 3:
            P(W3|S3) P(S3|S2)
            '''
            for x in range(0, len(sentence)):
                if x==0:
                    try:
                        initial_prob_value = self.ip[label[x]]
                    except:
                        initial_prob_value = 1e-10

                    try:
                        emission_value = self.ep[(sentence[x], label[x])]
                    except:
                        emission_value = 1e-10

                    nominator = math.log(emission_value * initial_prob_value)
                    wp = float(self.wcnt[sentence[x]]) / self.totalwordcount
                    if wp == 0.0:
                        wp = 1e-10
                    denom = math.log(wp)
                    result += nominator - denom
                else:
                    # P(W2|S2) P(S2|S1)
                    #
                    # P(W2) =
                    try:
                        tp_value = self.tp[(label[x], label[x-1])]
                    except:
                        tp_value = 1e-10
                    try:
                        emission_value = self.ep[(sentence[x], label[x])]
                    except:
                        emission_value = 1e-10
                    nominator = math.log(emission_value * tp_value)
                    wp = float(self.wcnt[sentence[x]]) / self.totalwordcount
                    if wp == 0.0:
                        wp = 1e-10
                    denom = math.log(wp)
                    result += nominator - denom
        elif model == "Complex":
            for x in range(0, len(sentence)):
                if x == 0:
                    # P(W1 | S1) = self.ep[(word, pos)]
                    # P(S1) = self.ip[pos]
                    try:
                        initial_prob_value = self.ip[label[x]]
                    except:
                        initial_prob_value = 1e-10
                    try:
                        emission_value = self.ep[(sentence[x], label[x])]
                    except:
                        emission_value = 1e-10

                    nominator = math.log(emission_value * initial_prob_value)

                    wp = float(self.wcnt[sentence[x]]) / self.totalwordcount

                    if wp == 0.0:
                        wp = 1e-10
                    denom = math.log(wp)
                    result += nominator - denom
                elif x ==1:
                    # P(W2|S2) P(S2|S1)
                    #
                    # P(W2) =
                    try:
                        tp_value = self.tp[(label[x], label[x - 1])]
                    except:
                        tp_value = 1e-10
                    try:
                        emission_value = self.ep[(sentence[x], label[x])]
                    except:
                        emission_value = 1e-10
                    nominator = math.log(emission_value * tp_value)
                    wp = float(self.wcnt[sentence[x]]) / self.totalwordcount
                    if wp == 0.0:
                        wp = 1e-10
                    denom = math.log(wp)
                    result += nominator - denom
                else:
                    # P(W3|S3) * P(S3|S2) * P(S3|S1)
                    try:
                        tp_value_1 = self.tp[(label[x], label[x - 1])]
                    except:
                        tp_value_1 = 1e-10
                    try:
                        tp_value_2 = self.tp[(label[x], label[x - 2])]
                    except:
                        tp_value_2 = 1e-10
                    try:
                        emission_value = self.ep[(sentence[x], label[x])]
                    except:
                        emission_value = 1e-10

                    nominator = math.log(emission_value * tp_value_1 * tp_value_2)
                    wp = float(self.wcnt[sentence[x]]) / self.totalwordcount
                    if wp == 0.0:
                        wp = 1e-10
                    denom = math.log(wp)
                    result += nominator - denom
        return result

    # Do the training!
    #
    def train(self, data):
        for d in data:
            # Calculate initial probability
            if self.ip[d[1][0]]:
                self.ip[d[1][0]] += 1
            else:
                self.ip[d[1][0]] = 1
            for i in range(1, len(d[1])):
                # Calculate transition probability
                if self.tp[(d[1][i], d[1][i-1])]:
                    self.tp[(d[1][i], d[1][i-1])] += 1
                else:
                    self.tp[(d[1][i], d[1][i-1])] = 1


            for idx, pos in enumerate(d[1]):

                if self.wcnt[d[0][idx]]:
                    self.wcnt[d[0][idx]] += 1
                else:
                    self.wcnt[d[0][idx]] = 1

                self.totalwordcount += 1
                if self.poscnt[pos]:
                    self.poscnt[pos] += 1
                else:
                    self.poscnt[pos] = 1
                # Calculate emission probability
                if self.ep[(d[0][idx], pos)]:
                    self.ep[(d[0][idx], pos)] += 1
                else:
                    self.ep[(d[0][idx], pos)] = 1

        self.ip = {k :float(v)/sum(self.ip.values()) for k, v in self.ip.items()}
        self.tp = {k:float(v)/self.poscnt[k[1]] for k, v in self.tp.items()}

        self.ep = {k: float(v) / self.poscnt[k[1]] for k, v in self.ep.items()}

        self.psprob = {k:float(self.poscnt[k]) / sum(self.poscnt.values()) for k, v in self.poscnt.items()}

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        # Calculate P(Si = si|Wi) = P(Wi|Si)P(Si) / P(Wi)
        postags = []
        for word in sentence:
            maxpsi = -sys.maxsize
            for pos, v in self.poscnt.items():
                try:
                    psi = self.ep[(word, pos)]
                except:
                    psi = 1e-10
                psi *= (float(self.poscnt[pos]) / sum(self.poscnt.values()))
                if psi > maxpsi:
                    maxpsi = psi
                    si = pos
            postags.append(si)
        return postags

    def gibbs_sampling(self, sample, sentence):
        #Factors P(W1|S1) P(W2|S2) P(W3|S3) P(S1) P(S2|S1) P(S3|S1,S2)... P(Wn|Sn)P(Sn|Sn-1,Sn-2)
        #Calculate P(Wi|Si)
        for idx, word in enumerate(sentence):
            prob_si = [0] * len(self.poscnt)
            pos_holder = []
            i = 0
            for pos, v in self.poscnt.items():
                pos_holder.append(pos)
                try:
                    emp = self.ep[(word, pos)]
                except:
                    emp = 1e-10
                try:
                    tpprev1 = self.tp[(pos, sample[idx-1])]
                except:
                    tpprev1 = 1e-10
                try:
                    tpprev2 = self.tp[(pos, sample[idx-2])]
                except:
                    tpprev2 = 1e-10

                if idx == 0:  #P(W1|S1) *  P(S1)
                    try:
                        inp = self.ip[pos]
                    except:
                        inp = 1e-10
                    prob_si[i] = emp * inp
                elif idx == 1:  #P(W2|S2) * P(S2|S1=(value assigned from previous iteration))
                    try:
                        prob_si[i] = emp * tpprev1
                    except:
                        prob_si[i] = 1e-10
                elif idx > 1:  #P(W3|S3) * P(S3|S2, S1)...P(Sn|Sn-1,Sn-2)
                    prob_si[i] = emp * tpprev1 * tpprev2
                i += 1
            prob_norm = [p/sum(prob_si) for p in prob_si]
            rand = random.random()
            sumofprob = 0
            for j, p in enumerate(prob_norm):
                sumofprob += p
                if rand < sumofprob:
                    sample[idx] = pos_holder[j]
                    break
        return sample


    def complex_mcmc(self, sentence):
        res = []
        #Intialize sample with noun for all words
        sample = ["noun"] * len(sentence)
        sample_generated = []
        for s in range(1000):
            sample_generated.append(self.gibbs_sampling(sample, sentence)[:])
        #converting list of list to dataframe
        df = pd.DataFrame(sample_generated)
        #finding max occurance in the dataframe columns
        for idx in range(len(sentence)):
            res.append(df[idx].value_counts().idxmax())
        return res

    def hmm_viterbi(self, sentence):
        for idx, word in enumerate(sentence):
            for pos in self.poscnt:
                vt_temp = {}
                try:
                    emp = self.ep[(word, pos)]
                except:
                    emp = 1e-10
                try:
                    inp = self.ip[pos]
                except:
                    inp = 1e-10

                if idx == 0:
                    # Calculate initial viterbi - ip(pos) * ep(word, pos)
                    self.hmmvt[str(pos)+str(idx)] = inp * emp
                else:
                    # Calculate rest of the viterbi - max{hmmvt(pos(idx-1)*tp[pos, pos],....}eppos(word)
                    for pos2 in self.poscnt:
                        try:
                            trp = self.tp[(pos, pos2)]
                        except:
                            trp = 1e-10
                        vt_temp[pos2] = self.hmmvt[str(pos2)+str((idx-1))] * trp
                    self.hmmvt[str(pos)+str(idx)+'max'] = max(vt_temp.items(), key=operator.itemgetter(1))[0]
                    self.hmmvt[str(pos)+str(idx)] = float( (max(vt_temp.items(), key=operator.itemgetter(1))[1]) )* emp
        res = []
        temp = {}
        j = 0
        for i in range((len(sentence)-1), -1, -1):
            if i == len(sentence)-1:
                temp = {pos3:self.hmmvt[str(pos3)+str(i)] for pos3 in self.poscnt}
                res.append( max(temp.items(), key=operator.itemgetter(1))[0] )
            else:
                res.append( self.hmmvt[ str(res[j-1])+str(i+1)+'max' ] )
            j += 1
        return res[::-1]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

