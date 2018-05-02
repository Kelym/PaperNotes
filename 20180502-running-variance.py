# Running Variance
# 
# While reading [PyTorch-RL code](https://github.com/Khrylx/PyTorch-RL) I found
# that they use a smart way of calculating variance which does not record
# all data points X. 
# 
# See a better explanation: 
#   http://www.johndcook.com/blog/standard_deviation/
# 
# See another implementation:
#   https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py

import numpy as np

class RunningStat:
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape) # Running mean
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        oldM = self._M.copy()
        self._n += 1
        self._M[...] = oldM + (x - oldM) / self._n
        self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else self._S

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

if __name__ == '__main__':
    rs = RunningStat([2])
    assert(all(rs.mean == np.zeros([2])))
    rs.push([1,2])
    rs.push([-2.7,-0.11])
    print("Mean = ", rs.mean)
    print("STD = ", rs.std)
