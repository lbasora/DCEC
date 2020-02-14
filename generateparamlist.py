import argparse
import numpy as np

SEED = 0
N = 1000

def testrandomgen():
    np.random.seed(SEED)
    '''generate all the random hyperparameters to test'''

    # number of layers
    nlayers = np.random.randint(1, 8, N)
    def generatearchirandom(sizelim=100,reverse=True):
        a = []
        possiblehidden = list(range(3, sizelim))
        for i in range(N):
            w = [np.random.randint(0, len(possiblehidden))
                 for _ in range(nlayers[i])]
            archi = "_".join([str(possiblehidden[x])
                              for x in sorted(w, reverse=reverse)])
            a.append(archi)
        return a

    def gen(size, channel):
        c = size * channel
        kernel = np.random.randint(1,size+1)
        nsize = size - kernel + 1
        nchannel = np.random.randint(c//nsize+1)
        return nsize, kernel, nchannel
    def gencnn1d(size,channel,nlayers):
        ks = []
        cs = []
        def aux(size, channel, nlayers):
            if nlayers == 0:
                return (ks,cs)
            else:
                nsize, kernel, nchannel = gen(size,channel)
                if nsize >=2:
                    ks.append(kernel)
                    cs.append(nchannel)
                    return aux(nsize,nchannel,nlayers-1)
                else:
                    return aux(size,channel,nlayers)
        return aux(size,channel,nlayers)

    def generatekernelfilterrandom(size=64,channel=4):
        ks = []
        cs = []
        for i in range(N):
            k,c = gencnn1d(size,channel,nlayers[i])
            ks.append("_".join([str(x) for x in k]))
            cs.append("_".join([str(x) for x in c]))
        return ks,cs

    r = {}
    optional = {}
    # architecture of the network (number of layers and number of hidden units)
    lmodels = ["dense","cae1d"]
    ks,cs=generatekernelfilterrandom()
    r["--model"] = ["dense"]*N#[lmodels[np.random.randint(len(lmodels))] for i in range(N)]
    r["--kernels"] = ks
    r["--archidense"] = generatearchirandom()
    r["--filters"] = cs
    return r
