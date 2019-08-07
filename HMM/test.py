import forward_algorithm
import viterbi_algorithm

observed_prob={
    'hot':{'1':0.2,'2':0.4,'3':0.4},
    'cold':{'1':0.5,'2':0.4,'3':0.1}
}

start_prob = {0:{'hot':0.8,'cold':0.2}}
hide_prob={
    'hot':{'hot':0.6,'cold':0.3,'end':0.1},
    'cold':{'hot':0.4,'cold':0.5,'end':0.1}
}

observed1=['3','1','3']
observed2=['3','3','1','1','2','2','3','1','3']

# forward_algorithm.forward_algorithm(observed_prob,hide_prob,observed2,start=start_prob)

probs = viterbi_algorithm.viterbi(observed_prob,hide_prob,observed1,start=start_prob)
print(probs)