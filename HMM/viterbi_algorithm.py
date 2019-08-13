from collections import defaultdict

''' 
O: observered event probablitics
H: hide params to hide params probablitics
oberserved: a serials of event
start: 
'''
def viterbi(O,H,oberserved,start):
    prev_probs=defaultdict(dict)

    def viterbi_impl(t,state):
        if prev_probs.__contains__(t):
            if prev_probs[t].__contains__(state):
                return prev_probs[t][state]

        if t==0:
            return start[0][state]*O[state][oberserved[t]]

        prob = max([viterbi_impl(t-1,pre_state)*H[pre_state][state]*O[state][oberserved[t]] for pre_state in H])
        prev_probs[t][state]=prob
        return prob
    
    probs=[]
    for t in range(len(oberserved)):
        probs.append(max([(state+'->'+oberserved[t],viterbi_impl(t,state)) for state in H.keys()],key=lambda x:x[1]))
    return probs