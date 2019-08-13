from collections import defaultdict

class Forward(object):
    def __init__(self,O,H,observed,start):
        self._prev_probs=defaultdict(dict)
        self._start=start
        self._O=O
        self._H=H
        self._observed=observed

    def forward_impl(self,t,state):
        if self._prev_probs.__contains__(t):
            if self._prev_probs[t].__contains__(state):
                return self._prev_probs[t][state]
        if t==0:
            # print(start[t][state])
            return self._start[state]*self._O[state][self._observed[t]]
        prob = sum([ self._H[prev_state][state]*self._O[state][self._observed[t]]*self.forward_impl(t-1,prev_state) for prev_state in self._H])
        self._prev_probs[t][self._observed[t]] = prob
        return prob

''' 
O: observered event probablitics
H: hide params to hide params probablitics
oberserved: a serials of event
start: 
'''
def forward_algorithm(O,H,oberserved,start):
    forward_probs=Forward(O,H,oberserved,start)

    for idx in range(len(oberserved)):
        for state in H.keys():
            print(state,'->',oberserved[idx],':',forward_probs.forward_impl(idx,state))
    
    # return prev_probs
