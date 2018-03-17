# ==========================================================================
# Tool for grading HMM
# --------------------------------------------------------------------------
# 
# ==========================================================================


states = ('R', 'G', 'Y') 
observations = ('Bright', 'Dark')
start_probability = {'R':0.3, 'G':0.4, 'Y':0.3}
#start_probability = {'R':0.5, 'G':0.5, 'Y':0}

transition_probability = {
    'R': {'R':0.8,'G':0.1, 'Y':0.1},
    'G': {'R':0.0,'G':0.3, 'Y':0.7},
    'Y': {'R':0.2,'G':0.5, 'Y':0.3}
}
emission_probability = {
    'R' : {'Bright':0.25, 'Dark':0.75},
    'G' : {'Bright':0.5, 'Dark':0.5},
    'Y' : {'Bright':0.8, 'Dark':0.2}
}

def print_dptable(prior, post):
    print("    ",end="")
    for i in range(len(prior)):
        print("%7d" % i,end="")
    print()
    print("Prior")

    for y in prior[0].keys():
        print("%6s: " % y,end="")
        for t in range(len(prior)):
            print("%7s" % ("%.4f" % prior[t][y]),end="")
        print()

    print("Post")

    for y in post[0].keys():
        print("%6s: " % y,end="")
        for t in range(len(post)):
            print("%7s" % ("%.4f" % post[t][y]),end="")
        print()

def hmm(observations, states, start_p, trans_p, emit_p):
    post = [{}]
    prior = [{}]

    for x in states:
        prior[0][x] = start_p[x]
        post[0][x] = start_p[x]

    t = 1
    for ob in observations:
        post.append({})
        prior.append({})

        for x in states:
            prior[t][x] = 0
            post[t][x] = 0

        for x in states:
            for y in states: 
                prior[t][y] += post[t-1][x] * trans_p[x][y]

        #prior[1]={'R':0.3, 'G':0.4, 'Y':0.3}
        #prior[1]={'R':0.31, 'G':0.35, 'Y':0.33}
        divider = 0
        for x in states:
            divider += prior[t][x] * emit_p[x][ob]

        for x in states:
            post[t][x] = prior[t][x] * emit_p[x][ob] / divider
        t = t+1

    print_dptable(prior, post)


hmm(observations,
      states,
      start_probability,
      transition_probability,
      emission_probability)