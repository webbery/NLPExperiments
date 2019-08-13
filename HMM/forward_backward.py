from collections import defaultdict
import random
from forward_algorithm import Forward
import numpy as np
import pandas as pd

class BackwardProbability(object):
    def __init__(self,O,H,oberserved,end):
        self._O=O
        self._H=H
        self._oberserved=oberserved
        self._probs=defaultdict(dict)
        self._end=end

    def prob(self,t,state):
        if t+1==len(self._oberserved): return self._end[state]
        # print(t,len(self._oberserved))
        if self._probs.__contains__(state):
            if self._probs[state].__contains__(t):
                return self._probs[state][t]
        p = sum([self._H[next_state_name][state]*self._O[next_state_name][self._oberserved[t+1]]*self.prob(t+1,next_state_name) for next_state_name in self._H ])
        self._probs[state][t] = p
        return p

def learn(oberserved,observe_states,hide_states):
    ''' E step '''
    ## random initialize A,B
    transition=defaultdict(dict)
    for state_name_from in hide_states:
        for state_name_to in hide_states:
            transition[state_name_from][state_name_to]=random.random()
    # print(transition)

    emition = defaultdict(dict)
    for state_name in hide_states:
        for o_name in observe_states:
            emition[state_name][o_name]=random.random()
    # print(emition)

    start=defaultdict(float)
    for state_name in hide_states:
        start[state_name]=random.random()
    # print(start)
    end=defaultdict(float)
    for state_name in hide_states:
        end[state_name]=random.random()

    forward = Forward(emition,transition,oberserved,start)
    backward = BackwardProbability(emition,transition,oberserved,end)

    ksi=pd.Panel(np.random.rand(len(oberserved)-1,len(hide_states),len(hide_states)),
        major_axis=hide_states,minor_axis=hide_states)
    ksi = ksi.to_frame()

    gamma = defaultdict(dict)
    P_O_lambda=defaultdict(float)

    for times in range(100):
        for t in range(len(oberserved)-1):
            P_O_lambda[t] = sum([forward.forward_impl(t,state)*backward.prob(t,state) for state in hide_states])
            # print(P_O_lambda[t])
            for state_i in hide_states:
                for state_j in hide_states:
                    ksi[t][state_i][state_j] = forward.forward_impl(t,state_i)*transition[state_i][state_j]*emition[state_j][oberserved[t+1]]*backward.prob(t+1,state_j)/P_O_lambda[t]
                gamma[t][state_i]=forward.forward_impl(t,state_i)*backward.prob(t,state_i)/P_O_lambda[t]
        # print(emition)
        ''' M step '''
        ## learning and update new params of A,B
        for state_i in hide_states:
            for state_j in hide_states:
                numerator = sum([ ksi[idx][state_i][state_j] for idx in range(len(oberserved)-1)])
                sum_ksi = sum([ sum([ ksi[idx][state_i][state_k] for state_k in hide_states]) for idx in range(len(oberserved)-1)])
                transition[state_i][state_j]=numerator/sum_ksi
            for obs in observe_states:
                sum_gammar = 0
                for t in range(len(oberserved)-1):
                    # print(obs,sum_gammar,oberserved[t])
                    if oberserved[t]==obs: sum_gammar+=gamma[t][state_i]
                emition[state_i][obs]=sum_gammar/sum([gamma[t][state_i] for t in range(len(oberserved)-1)])
        if times%5==0:
            print(transition)
            print(emition)
    
