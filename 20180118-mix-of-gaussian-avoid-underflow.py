'''
Expectation Maximization
------------------------
To run, launch 
    python expect-maximize.py <# of clusters> <data file> <model file>
    python expect-maximize.py vehicle.train model.tmp

The data file looks like
    <# of examples> <# of features>
    <ex.1, feature 1> <ex.1, feature 2> … <ex.1, feature n> <ex.1, label>
    <ex.2, feature 1> <ex.2, feature 2> … <ex.2, feature n> <ex.2, label>

The output model file looks like
    <# of clusters> <# of features>
    <clust1.prior> <clust1.mean1> <clust1.mean2> … <clust1.var1> … 
    <clust2.prior> <clust2.mean1> <clust2.mean2> … <clust2.var1> …

This version calculates probability in its log form to avoid underflow of float.
'''
import numpy as np
from scipy.stats import norm

# To avoid np.log(x) warning of divide by 0, we can add a small amount to x
ABS = 1e-15

def log_sum_exp(x):
    # Handle log \sum exp(x_i) = x_max + log \sum exp(x_i - x_max)
    # Input x of shape (N, D)
    # Return (D)
    return (np.max(x, axis=0) + 
        np.log(np.sum(np.exp(x - np.max(x, axis=0)), axis=0)))

def test_log_sum_exp(x):
    # Test the log_sum_exp
    res = np.log(np.sum(np.exp(x), axis=0))
    res2 = log_sum_exp(x)
    return (res - res2)

def logexpectation(x, priors, means, sigmas, normalize=True):
    # X are the data of shape [N, D]
    # priors are the prior of models, shape [K]
    # Posterior = [N, K], prob of x_i coming from cluster k, normalized and log
    N, D = x.shape
    K = priors.shape[0]
    priors = np.log(priors)
    posterior = np.zeros((N, K))
    for i in range(K):
        posterior[:, i] = priors[i] + \
            np.sum(norm.logpdf(x, means[i], sigmas[i]), axis=1)
    if normalize: posterior = posterior - log_sum_exp(posterior.T)[:, None]
    return posterior

def maximization(x, logposterior):
    # Given log of posterior prob, find the best model
    N, K = logposterior.shape
    _, D = x.shape
    sum_posterior_k = log_sum_exp(logposterior)           # [K]
    priors = np.exp(sum_posterior_k - np.log(N))          # [K]
    fx = np.expand_dims(np.log(x + ABS), axis=1)                # [N, 1, D]
    fp = np.expand_dims(logposterior, axis=2)             # [N, K, 1]
    means = np.exp(log_sum_exp(fx + fp) - sum_posterior_k[:, None]) # [K,D]
    posterior = np.exp(logposterior)
    dx = np.expand_dims(x, axis=1) - np.expand_dims(means, axis=0)  # [N, K, D]
    dx = 2 * np.log(np.abs(dx) + ABS)
    sigmas = np.exp(0.5 * (log_sum_exp(dx + fp) - sum_posterior_k[:, None])) # [K,D]
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
    logposterior = logexpectation(x, priors, means, sigmas, normalize=False)
    loglikelihoods = log_sum_exp(logposterior.T)
    return np.sum(loglikelihoods)

def evaluate_labels(x, priors, means, sigmas, labels, cluster_label=None):
    N, D = x.shape
    K = priors.shape[0]
    TK = np.unique(labels)
    count_correct = 0
    logposterior = logexpectation(x, priors, means, sigmas)
    label = np.argmax(logposterior, axis=1)
    if cluster_label is not None:
        label = cluster_label[label]
        count_correct = (labels == label).sum()
    else:
        cluster_label = []
        for i in range(K):
            bucket = labels[label==i]
            unique, pos = np.unique(bucket, return_inverse=True)
            counts = np.bincount(pos)
            maxpos = counts.argmax()
            count_correct += counts[maxpos]
            cluster_label.append(unique[maxpos])
        cluster_label = np.array(cluster_label)
    return count_correct / float(N), cluster_label

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('num_cluster', type=int)
    parser.add_argument('data_file', type=str)
    parser.add_argument('model_file', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--init_uniform', action='store_true')
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    N, D, x, label = load_data(args.data_file, hasLabel=True)
    if args.test_file:
        _, _, x_test, label_test = load_data(args.test_file, hasLabel=True)

    # Train EM
    print("Training MoGaussian model with %d clusters, init to %s" % 
            (args.num_cluster, 'uniform' if args.init_uniform else 'data'))
    priors, means, sigmas = init_model(args.num_cluster, x, 
                                        uniform=args.init_uniform)
    loghis = [-float("inf")]
    loghis_t = []
    for itr in range(1,10000):
        p = logexpectation(x, priors, means, sigmas)
        priors, means, sigmas = maximization(x, p)

        loghis.append(loglikelihood(x, priors, means, sigmas))
        print(loghis[-1])
        if args.test_file: loghis_t.append(loglikelihood(x_test, priors, means, sigmas))
        
        if args.verbose: print("== Iter %d \t Train LogLike %.6f" % (itr, loghis[-1]))
        if loghis[-2] * 0.999 >= loghis[-1]: break
        if np.isnan(loghis[-1]):
            # Cluster collapse .. 
            import sys
            sys.exit(0)

    loghis.pop(0)
    print("Convergence took %d iters" % itr)
    print("Final training log likelihood %f" % loghis[-1])

    # Save Model
    save_model(args.model_file, priors, means, sigmas)

    # Evaluate
    train_perf, cluster_label = evaluate_labels(x, priors, means, sigmas, label)
    print("Accuracy on training set (compared with gt) %.6f " % train_perf)
    if args.test_file:
        test_perf, _ = evaluate_labels(x_test, priors, means, sigmas, label_test, cluster_label)
        print("Accuracy on test set (compared with gt) %.6f " % test_perf)

    if args.log_file:
        with open(args.log_file, "a+") as f:
            # Num cluster, Init, Seed, Iter, Loglikelihood, Accuracy
            f.write("%d %d %d %d %.8f %.3f" % (args.num_cluster, 
                args.init_uniform, args.seed, itr, loghis[-1], train_perf))
            if args.test_file:
                f.write(" %.8f %.3f\n" % (loghis_t[-1], test_perf))
            else:
                f.write("\n")

if __name__ == '__main__':
    main()
