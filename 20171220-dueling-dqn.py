# ==========================================================================
# Dueling DQN, compared with DQN, on OpenAI gym CartPole.
# --------------------------------------------------------------------------
# Code adapted from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5.3_Dueling_DQN
#
# Practice Deep Reinforcement Learning with Tensorflow
#
# Deep Q Network is a way to do Reinforcement Learning. Specifically, it 
#   uses a neural network to approximate the expected reward of taking 
#   action a at state s, denoted as Q(s,a). Then, when playing a game, 
#   the player could choose the action that maximize the expected reward
#   for the current state. Q is usually calculated by:
#       Q(s, a) = reward + reward_decay * max[Q(s_, a_)]
#           where max[Q(s_, a_)] is the best reward for the next state s_
# 
# Double DQN tries to avoid overestimating max[Q(s_, a_)]. This might in 
#   DQN because it uses the same Q value to select and evaluate an action.
#   In double DQN, we copy the Q value to Q_next and calculate
#   max[Q_next(s_, a_)]. While updating Q values, we freeze the Q_next.
#   We only copy Q value to Q_next at a given frequency. 
#
# Dueling Q Network considers Q(s, a) = V(s) + A(s,a) where
#   i) V is the "value" of a state and is independent from whatever action
#   ii) A(s,a) is the "advantage" of taking action a at state s. 
#       The advantage is calculated relative to all possible actions at s. 
#
# This code compares performance of DQN and DuelingQN, while both uses
#   double Q network, on a OpenAI Gym cartpole game. 
#  
# ==========================================================================
# 
import gym
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 

np.random.seed(13)
tf.set_random_seed(13)
env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(7)
ACTION_SPACE = 25 # Discretize the continuous action space of cartpole
MEMORY_SIZE = 3000
train_iter = 20000
train_freq = 10

def main():
    sess = tf.Session()
    with tf.variable_scope('natural'):
        natural_DQN = DuelingDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=None, sess=sess, dueling=False)
    with tf.variable_scope('dueling'):
        dueling_DQN = DuelingDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=None, sess=sess, dueling=True)
    sess.run(tf.global_variables_initializer())
    l_natural, r_natural = train(natural_DQN)
    l_dueling, r_dueling = train(dueling_DQN)

    plt.figure(1)
    plt.plot(np.array(l_natural), c='r', label='natural')
    plt.plot(np.array(l_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('Loss')
    plt.xlabel('training steps')
    plt.grid()

    plt.figure(2)
    plt.plot(np.array(r_natural), c='r', label='natural')
    plt.plot(np.array(r_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()

    plt.show()

def train(RL):
    acc_r = [0]
    observation = env.reset()
    for step in range(train_iter):
        action = RL.choose_action(observation)
        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
        observation_, reward, _, _= env.step(np.array([f_action]))
        acc_r.append(reward + acc_r[-1])
        RL.store_transition(observation, action, reward, observation_)
        observation = observation_
        if ( step % train_freq == 0 and
            step > RL.batch_size): RL.learn()
        #if step % 100 == 0: env.render()
    return RL.loss_his, acc_r

class DuelingDQN:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=1e-3,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200, # Update Target Network Value
        memory_size=500,
        batch_size=100,
        e_greedy_increment=None,
        dueling=True,
        sess=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dueling = dueling
        self.replace_target_iter = replace_target_iter
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy if e_greedy_increment is None else 0
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2)) #For [s,[a,r],s_]
        self.memory_counter = 0
        self.loss_his = []
        self.sess = sess
        self._build_net()

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_init, b_init):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    # Q = V(s) + A(s,a)
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # State
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')

        # Placeholder for the desired Q value
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        # Evaluate expected reward for the current state
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        # Define loss and training operation
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Every training step we are changing q_eval, thus q_eval can be unstable
        # Instead of letting the next state chooses the action that maximize q_eval,
        # Copy q_eval to q_next at a certain frequency, and let the next state
        # choose the action that maximize q_next
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('copy_net'):
            c_names = ['copy_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

        # Update q_eval and q_next at a frequency
        t_params = tf.get_collection('copy_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def unpack_transition(self, index):
        batch_memory = self.memory[index, :]
        s = batch_memory[:, :self.n_features]
        s_ = batch_memory[:, -self.n_features:]
        a = batch_memory[:, self.n_features].astype(int)
        r = batch_memory[:, self.n_features + 1]
        return (s, a, r, s_)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('q_next_replaced')

        # Select a batch
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        s, a, r, s_ = self.unpack_transition(sample_index)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_next = self.sess.run(self.q_next, feed_dict={self.s_: s_})
        q_eval = self.sess.run(self.q_eval, {self.s: s})

        q_target = q_eval.copy()
        q_target[batch_index, a] = r + self.reward_decay * np.max(q_next, axis=1)

        _, loss = self.sess.run([self._train_op, self.loss], 
                                 feed_dict={self.s: s, self.q_target: q_target})
        self.loss_his.append(loss)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

if __name__ == '__main__': main()

