# ==========================================================================
# Have fun with Viterbi algorithm
# --------------------------------------------------------------------------
# Reference: http://en.wikipedia.org/wiki/Viterbi_algorithm
# 
# Viterbi algorithm is a dynamic programming algorithm for finding the most 
# likely sequence of hidden states—called the Viterbi path—that results in a 
# sequence of observed events. 
# 
# It assumes that the world has several possible states. However, the algorithm
# cannot see the state directly. At each time period, the algorithm makes
# observations of the world. The probability for each state s to emit 
# observation o is P(o,s). The world evolves in each time period such that the
# probability to transit from state s1 to s2 is P(s1->s2). 
# At the begining of observation, each state is assigned a likelihood (start 
# probability) P_start(s).
# 
# Given all possible states and their start possibilities, observation history, 
# transition probability and emission probability, the algorithm could find
# the most likely state history of the world's evolution. 
# ==========================================================================


states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy':0.6, 'Fever':0.4}
transition_probability = {
    'Healthy' : {'Healthy' : 0.7, 'Fever' : 0.3},
    'Fever' : {'Healthy' : 0.4, 'Fever' : 0.6}
}
emission_probability = {
    'Healthy' : {'normal' : 0.5, 'cold' : 0.4, 'dizzy' : 0.1},
    'Fever' : {'normal' : 0.1, 'cold' : 0.3, 'dizzy' : 0.6}
}

def print_dptable(V):
    print("    ",end="")
    for i in range(len(V)):
        print("%7d" % i,end="")
    print()

    for y in V[0].keys():
        print("%.5s: " % y,end="")
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]),end="")
        print()

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
        
            V[t][y] = prob
            new_path[y] = path[state] + [y]

        path = new_path

    print_dptable(V)
    (prob, state) = max([(V[-1][y], y) for y in states])
    return (prob, path[state])


print(viterbi(observations,
              states,
              start_probability,
              transition_probability,
              emission_probability))