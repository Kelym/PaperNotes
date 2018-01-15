# ==========================================================================
# Policy Gradient in tensorflow
# --------------------------------------------------------------------------
# Cleaned my solution to Sergey Levine UCB Course Deep RL, HW2
# 
# Policy Gradient is a way to do Reinforcement Learning. Given a state, it 
#   tries to output the action directly. Compared with Q learning which
#   evaluates the expected reward of taking certain action at a given 
#   state, policy gradient can handle continuous action space easily. 
#
# For discrete action spaces, th e neural net reads the current state and 
#   output an unnormalized log probability for each action. For continuous
#   action spaces, the net outputs the mean and variance for each dimension
#   of the action space. To fit the neural net, we sample trajectories from
#   the net, observe the rewards obtained, and encourage or discourage
#   actions by their "advantages", e.g. accumulated discount reward for the
#   trajectory. 
#
# This code offers two optional tricks: reward-to-go and baseline. 
#   - Reward to go: assign "advantage" of each action to be the accumulated
#       discounted reward **AFTER** that action is executed. 
#   - Baseline: use a neural network to predict the reward for any state. 
#       Normalize the predicted rewards in a batch and scale to have the same
#       variance as the accumulated rewards observed. This is the baseline. 
#       The advantage of action is now the difference between the baseline
#       and the accumulated reward. 
# 
# To run:
#   python 20171221-clean-policy-gradient.py 'CartPole-v1'
# ==========================================================================

import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

# Suppress CPU instruction set warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(input_placeholder, output_size, scope, n_layers=2, size=64,
    activation=tf.tanh, output_activation=None):
    with tf.variable_scope(scope):
        dense = input_placeholder 
        for i in range(n_layers):
            dense = tf.layers.dense(
                        inputs=dense, 
                        units=size,
                        activation=activation)
        return tf.layers.dense(
            inputs=dense, 
            units=output_size, 
            activation=output_activation)

def pathlength(path): return len(path["reward"])

#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             n_layers=1,
             size=32
             ):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Placeholders
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) 

    if discrete:
        sy_logits_na = build_mlp( 
            input_placeholder=sy_ob_no,
            output_size=ac_dim,
            scope="build_nn",
            n_layers=n_layers,
            size=size,
            activation=tf.nn.relu) # The output should be an unnormalized log-prob of action
        print(sy_logits_na.shape)
        sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1), axis=[1]) # sampled ac
        sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sy_ac_na, 
            logits=sy_logits_na)

    else:
        sy_mean = build_mlp(
            input_placeholder=sy_ob_no,
            output_size=ac_dim,
            scope="build_nn",
            n_layers=n_layers,
            size=size,
            activation=tf.nn.relu)
        sy_logstd = tf.get_variable("logstd",shape=[ac_dim]) 
        sy_sampled_ac = sy_mean + tf.multiply(tf.exp(sy_logstd),
                                              tf.random_normal(tf.shape(sy_mean)))
        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_mean, 
            scale_diag=tf.exp(sy_logstd)) 
        sy_logprob_n = -dist.log_prob(sy_ac_na)

    weighted_negative_likelihood = tf.multiply(sy_logprob_n, sy_adv_n)
    loss = tf.reduce_mean(weighted_negative_likelihood)
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                input_placeholder=sy_ob_no, 
                                output_size=1, 
                                scope="nn_baseline",
                                n_layers=n_layers,
                                size=size))
        baseline_target = tf.placeholder(shape=[None], dtype=tf.float32)
        baseline_loss = tf.losses.mean_squared_error(predictions=baseline_prediction, labels=baseline_target)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)


    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                ac = ac[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        # Calculate accumulated discount reward
        def discount_rewards_to_go(rewards, gamma):
            res = [] 
            future_reward = 0
            for r in reversed(rewards):
                future_reward = future_reward * gamma + r
                res.append(future_reward)
            return res[::-1]

        def sum_discount_rewards(rewards, gamma):
            return sum((gamma**i) * rewards[i] for i in range(len(rewards)))

        q_n = []
        if reward_to_go:
            q_n = np.concatenate([discount_rewards_to_go(path["reward"], gamma) for path in paths])
        else:
            q_n = np.concatenate([
                    [sum_discount_rewards(path["reward"], gamma)] * pathlength(path)
                    for path in paths])

        if nn_baseline and itr > 0:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no: ob_no})
            b_n = (b_n - np.mean(b_n)) / (np.std(b_n)+1e-10) * np.std(q_n) + np.mean(q_n)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        # Advantage Normalization
        if normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n)+1e-10)


        if nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            scaled_q = (q_n - np.mean(q_n)) / (np.std(q_n) + 1e-10)
            _ = sess.run(baseline_update_op, feed_dict={
                sy_ob_no : ob_no,
                baseline_target: scaled_q})

        _, loss_value = sess.run([update_op, loss], feed_dict={sy_ob_no: ob_no,
            sy_ac_na: ac_na,sy_adv_n: adv_n})

        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        print("AverageReturn", np.mean(returns))
        print("StdReturn", np.std(returns))
        print("MaxReturn", np.max(returns))
        print("MinReturn", np.min(returns))
        print("EpLenMean", np.mean(ep_lengths))
        print("EpLenStd", np.std(ep_lengths))
        print("TimestepsThisBatch", timesteps_this_batch)
        print("TimestepsSoFar", total_timesteps)
        print("Loss", loss_value)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')): os.makedirs('data')

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()

if __name__ == "__main__":
    main()
