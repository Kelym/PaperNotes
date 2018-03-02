'''
Expectation Maximization
------------------------
To run, launch 
    python expect-maximize.py <# of clusters> <data file> <model file>
The data file looks like
    <# of examples> <# of features>
    <ex.1, feature 1> <ex.1, feature 2> … <ex.1, feature n> <ex.1, label>
    <ex.2, feature 1> <ex.2, feature 2> … <ex.2, feature n> <ex.2, label>
The output model file looks like
    <# of clusters> <# of features>
    <clust1.prior> <clust1.mean1> <clust1.mean2> … <clust1.var1> … 
    <clust2.prior> <clust2.mean1> <clust2.mean2> … <clust2.var1> …
'''
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def expectation(x, priors, means, sigmas, normalize=True):
    # X are the data of shape [N, D]
    # priors are the prior of models, shape [K]
    # Posterior = [N, K], prob of x_i coming from cluster k, normalized
    N, D = x.shape
    K = priors.shape[0]
    posterior = np.zeros((N, K))
    for i in range(K):
        posterior[:, i] = priors[i] * np.prod(norm.pdf(x, means[i], sigmas[i]), axis=1)
        #Alternatively, multivariate_normal.pdf(x, mean=means[i], cov=np.diag(sigmas[i]**2))
    if normalize: posterior /=  np.sum(posterior, axis=1, keepdims=True)
    return posterior

def maximization(x, posterior):
    N, K = posterior.shape
    _, D = x.shape
    sum_posterior_k = np.sum(posterior, axis=0)           # [K]
    priors = sum_posterior_k / float(N)                   # [K]
    means = posterior.T.dot(x) / sum_posterior_k[:, None] # [K,D]
    sigmas = np.zeros((K, D))
    for i in range(K):
        devia = (x - means[i, :])**2                      # (N, D)
        sigmas[i, :] = posterior[:, i].dot(devia) / sum_posterior_k[i]
    sigmas = np.sqrt(sigmas)
    return priors, means, sigmas

def init_model(num_cluster, x, uniform):
    N, D = x.shape
    lb = np.min(x, axis=0)
    ub = np.max(x, axis=0)
    lu = ub - lb
    priors = np.array([1 / float(num_cluster)] * num_cluster)
    sigmas = np.array([lu / np.sqrt(num_cluster)] * num_cluster)
    if uniform:
        means = np.random.random((num_cluster, D)) * lu + lb
    else: # select data point to be used as cluster mean
        index = np.random.choice(range(N), num_cluster, replace=False)
        means = x[index]
    return priors, means, sigmas

def load_data(data_file, hasLabel=True):
    with open(data_file, 'r') as f:
        N, D = f.readline().strip().split()
        N, D = int(N), int(D)
        x = []
        label = []
        for i in range(N):
            line = f.readline().strip().split()
            if hasLabel: label.append(line.pop(-1))
            x.append([float(a) for a in line])
    return N, D, np.array(x), np.array(label)

def save_model(model_file, priors, means, sigmas):
    # Save a mix of Gaussian model to model_file
    # <# of clusters> <# of features>
    # <clust1.prior> <clust1.mean1> <clust1.mean2> … <clust1.var1> … 
    # <clust2.prior> <clust2.mean1> <clust2.mean2> … <clust2.var1> …
    with open(model_file, 'w') as f:
        K, D = means.shape
        f.write("%d %d\n" % (K, D))
        for i in range(K):
            f.write("%d %s %s\n" % (priors[i], ' '.join(map(str, means[i])),
                                    ' '.join(map(str, sigmas[i]**2))))

def loglikelihood(x, priors, means, sigmas):
    posterior = expectation(x, priors, means, sigmas, normalize=False)
    likelihood = np.sum(posterior, axis=1, keepdims=True)
    return np.sum(np.log(likelihood))

def evaluate_labels(x, priors, means, sigmas, labels):
    N, D = x.shape
    K = priors.shape[0]
    TK = np.unique(labels)

    count_correct = 0
    
    posterior = expectation(x, priors, means, sigmas)
    label = np.argmax(posterior, axis=1)
    for i in range(K):
        bucket = labels[label==i]
        unique, pos = np.unique(bucket, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        count_correct += counts[maxpos]

    return count_correct / float(N)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('num_cluster', type=int)
    parser.add_argument('data_file', type=str)
    parser.add_argument('model_file', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_file', type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    N, D, x, label = load_data(args.data_file, hasLabel=True)
    if args.test_file:
        _, _, x_test, label_test = load_data(args.test_file, hasLabel=True)

    # Train EM
    priors, means, sigmas = init_model(args.num_cluster, x, uniform=True)
    loghis = [-float("inf")]
    loghis_t = []
    for itr in range(1,10000):
        p = expectation(x, priors, means, sigmas)
        priors, means, sigmas = maximization(x, p)

        loghis.append(loglikelihood(x, priors, means, sigmas))
        print(loghis[-1])
        if args.test_file: loghis_t.append(loglikelihood(x_test, priors, means, sigmas))
        
        if loghis[-2] * 0.999 >= loghis[-1]: break
    loghis.pop(0)
    print("Convergence took %d iters" % itr)

    # Plot
    plt.plot(loghis, label='Train')
    if args.test_file: plt.plot(loghis_t, label='Test')
    plt.ylabel('Log Likelihood')
    plt.legend(loc='lower right', shadow=True)
    plt.show()

    # Evaluate
    train_perf = evaluate_labels(x, priors, means, sigmas, label)
    print("Accuracy on training set (compared with gt) %.6f " % train_perf)

    save_model(args.model_file, priors, means, sigmas)

    if args.test_file:
        test_perf = evaluate_labels(x_test, priors, means, sigmas, label_test)
        print("Accuracy on test set (compared with gt) %.6f " % test_perf)

if __name__ == '__main__':
    main()