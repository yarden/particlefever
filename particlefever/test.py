import numpy as np

def s(l):
    m = np.max(l)
    norm = m + np.log(np.sum(np.exp(l - m)))
    p = np.exp(l - norm)
    return np.where(np.random.multinomial(1, p) == 1)[0][0]

def regular_sample(l):
    return np.where(np.random.multinomial(1, l) == 1)[0][0]

def comp(iters=1000):
    logsamp = []
    naive = []
#    l = [0.2, 0.8]
    l = [-800, -1600]
    print "sampling from: ", l
    for n in range(iters):
        logsamp.append(s(l))
        naive.append(regular_sample(np.exp(l)))
    print "---"
    print "Percentage of 0"
    logsamp = np.array(logsamp)
    naive = np.array(naive)
    no_logsamp = len(np.where(logsamp == 0)[0])
    no_naive = len(np.where(naive == 0)[0])
    print "logsamp: %.3f" %(no_logsamp / float(iters))
    print "naive: %.3f" %(no_naive / float(iters))


comp()
