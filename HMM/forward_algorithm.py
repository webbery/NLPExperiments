from collections import defaultdict

''' 
O: observered event probablitics
H: hide params to hide params probablitics
oberserved: a serials of event
start: 
'''
def forward_algorithm(O,H,oberserved,start):
    prev_probs=defaultdict(dict)
    def forward_impl(t,state):
        if prev_probs.__contains__(t):
            if prev_probs[t].__contains__(state):
                return prev_probs[t][state]
        if t==0:
            # print(start[t][state])
            return start[0][state]*O[state][oberserved[t]]
        prob = sum([ H[prev_state][state]*O[state][oberserved[t]]*forward_impl(t-1,prev_state) for prev_state in H])
        prev_probs[t][oberserved[t]] = prob
        return prob

    for idx in range(len(oberserved)):
        for state in H.keys():
            print(state,'->',oberserved[idx],':',forward_impl(idx,state))
    
    # return prev_probs
