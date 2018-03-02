# ==========================================================================
# Policy Gradient in pytorch
# --------------------------------------------------------------------------
# 
# Policy Gradient is a way to do Reinforcement Learning. Given a state, it 
#   tries to output the action directly. Compared with Q learning which
#   evaluates the expected reward of taking certain action at a given 
#   state, policy gradient can handle continuous action space easily. 
#
# For discrete action spaces, the neural net reads the current state and 
#   output an unnormalized log probability for each action. For continuous
#   action spaces, the net outputs the mean for each dimension of the action
#   space. To fit the neural net, we sample trajectories from the net, observe
#   the rewards obtained, and encourage or discourage actions by their
#   "advantages". The advantage can be the accumulated discount reward for the
#   trajectory, or other things. 
#
# This code uses reward-to-go and baseline. 
#
# To run:
#   python 20180115-policy-gradient-torch.py
# ==========================================================================

import gym
import numpy as np 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--max_path_length', type=float, default=-1.)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.99)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--cuda', '-c', action='store_true')
    return parser

class Policy(nn.Module):
    # Multi layer perceptron
    def __init__(self, env, nn_sizes, activation=nn.ReLU()):
        super(Policy, self).__init__()
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        self.hidden_layers = nn.ModuleList()
        s = [ob_dim] + nn_sizes
        for i in range(len(s) - 1):
            self.hidden_layers.append(nn.Linear(s[i], s[i+1]))
            self.hidden_layers.append(activation)
        self.output = nn.Linear(nn_sizes[-1], ac_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output(x)

class PG:
    # Policy Gradient Agent
    def __init__(self, env, nn_sizes, lr, weight_decay, gamma, cuda):
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.model = Policy(env, nn_sizes, activation=nn.ReLU())
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, 
                                             weight_decay=weight_decay)
        self.cuda = cuda
        if cuda: self.model.cuda()

        if not self.discrete:
            self.logstd = Variable(torch.randn(1, self.ac_dim), 
                                    requires_grad=True)
            if cuda: self.logstd.cuda()

    def read_model(self, filename):
        import os
        if os.path.isfile(filename):
            print('Load Policy Gradient Torch param from ', filename)
            self.model.load_state_dict(torch.load(filename))
        else:
            print('Cannot find existing params at ', filename)

    def select_action(self, state):
        # Given a single state, return a sampled action and the log_prob
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        if self.cuda: state.cuda()
        scores = self.model(state)
        if self.discrete:
            m = torch.distributions.Categorical(F.softmax(scores, dim=1))
            action = m.sample()
            prob = m.log_prob(action)
        else:
            # Not working yet ... 
            # PyTorch don't propagate gradients through distributions ...
            # Let's go back to Tensorflow!
            normal_dist = torch.distributions.Normal(scores, torch.exp(self.logstd))
            action = normal_dist.sample()
            prob = normal_dist.log_prob(action)
        return action.data[0], prob

    @staticmethod
    def reward_to_go(rewards, gamma):
        rtg = []
        acc_r = 0
        for r in reversed(rewards):
            acc_r = acc_r * gamma + r
            rtg.append(acc_r)
        return rtg[::-1]

    def update(self, paths):
        # Calculate advantage from memory
        rewards = np.concatenate([PG.reward_to_go(path["reward"], self.gamma) 
                                    for path in paths])
        rewards = torch.Tensor(rewards)
        if self.cuda: rewards.cuda()
        rewards = (rewards - rewards.mean()) / \
                    (rewards.std() + np.finfo(np.float32).eps)
        
        # Calculate loss from advantage
        log_probs = [prob for path in paths for prob in path["log_prob"]]
        policy_loss = [ (- log_prob * reward) 
                        for log_prob, reward in zip(log_probs, rewards)]
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        print(self.logstd.data)
        print(self.output)
        return loss.data[0]

def train_PG(
    env_name,
    max_path_length,                # max trajectory length
    n_iter,                         # of training iter
    gamma,                          # decay rate of reward in future
    min_timesteps_per_batch,        # batch size
    lr,                             # learning rate
    weight_decay,                   # weight decay of learner
    reward_to_go,                   # whether to use reward to go
    animate,                        # whether to render
    nn_baseline,                    # whether to use baseline
    seed,                           # init random with seed
    nn_sizes,                       # net structure
    cuda
    ):
    # Set env
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    if cuda: torch.cuda.manual_seed(args.seed)

    # Restrict max episode length
    max_path_length = max_path_length if max_path_length > 0 else None
    max_path_length = max_path_length or env.spec.max_episode_steps
    
    # Build agent
    agent = PG(env, nn_sizes, lr, weight_decay, gamma, cuda)
    agent.read_model('tmp/pg_torch_params.pkl')

    # Training
    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, log_probs, rewards = [], [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac, prob = agent.select_action(ob)
                acs.append(ac)
                log_probs.append(prob)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length: break
            paths.append({"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : acs,
                    "log_prob": log_probs})
            timesteps_this_batch += steps
            if timesteps_this_batch > min_timesteps_per_batch: break

        total_timesteps += timesteps_this_batch
        loss_value = agent.update(paths)
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [path["reward"].shape[0] for path in paths]
        print("AverageReturn", np.mean(returns))
        print("StdReturn", np.std(returns))
        print("MaxReturn", np.max(returns))
        print("MinReturn", np.min(returns))
        print("EpLenMean", np.mean(ep_lengths))
        print("EpLenStd", np.std(ep_lengths))
        print("TimestepsThisBatch", timesteps_this_batch)
        print("TimestepsSoFar", total_timesteps)
        print("Loss", loss_value)

if __name__ == "__main__":
    args = get_parser().parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    train_PG(
        env_name=args.env_name,
        max_path_length=args.max_path_length,
        n_iter=args.n_iter,
        gamma=args.discount,
        min_timesteps_per_batch=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        reward_to_go=args.reward_to_go,
        animate=args.render,
        nn_baseline=args.nn_baseline, 
        seed=args.seed,
        nn_sizes=[args.size]*args.n_layers,
        cuda=args.cuda
        )
