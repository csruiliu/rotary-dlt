import numpy as np
import tensorflow as tf
import gym


class PolicyGradient:
    def __init__(self, n_feature, n_action_space, n_action_output, learning_rate=0.01):
        self.num_feature = n_feature
        self.num_action_space = n_action_space
        self.num_action_output = n_action_output
        self.learn_rate = learning_rate
        self.policy_logit = None
        self.policy_loss_op = None
        self.policy_train_op = None
        self.policy_action = None

    # alternative mlp model
    def build_policy_network_alternative(self, obs_batch, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network.
        for size in sizes[:-1]:
            obs_batch = tf.layers.dense(obs_batch, units=size, activation=activation)
        self.policy_logit = tf.layers.dense(obs_batch, units=sizes[-1], activation=output_activation)

        return self.policy_logit

    # the input of this function is a batch of observation and compute a batch of actions
    def build_policy_network(self, obs_batch):
        with tf.variable_scope('policy_network'):
            hidden_layer_neurons = 10
            variable_initializer = tf.contrib.layers.xavier_initializer()
            W1 = tf.get_variable("W1", shape=[self.num_feature, hidden_layer_neurons], initializer=variable_initializer)
            hidden_layer = tf.nn.relu(tf.matmul(obs_batch, W1))
            W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, self.num_action_space], initializer=variable_initializer)
            # generate the policy network model logit
            self.policy_logit = tf.matmul(hidden_layer, W2)

        return self.policy_logit

    # the input of this function is a batch of actions
    def train_policy_network(self, action_batch_ph, weights_batch_ph):
        # transferring the batch of actions to onehot
        actions_onehot = tf.one_hot(action_batch_ph, _num_action_space)
        log_probs = tf.reduce_sum(actions_onehot * tf.nn.log_softmax(self.policy_logit), axis=1)
        self.policy_loss_op = -tf.reduce_mean(weights_batch_ph * log_probs)
        self.policy_train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.policy_loss_op)

        return self.policy_loss_op, self.policy_train_op

    # use the policy network model to generate actions
    def action_policy_network(self):
        self.policy_action = tf.squeeze(tf.multinomial(logits=self.policy_logit, num_samples=self.num_action_output))

        return self.policy_action


def test_random_action(total_episodes):
    _env.reset()
    cur_episodes = 0
    reward_sum = 0
    while cur_episodes < total_episodes:
        _env.render()
        observation, reward, done, info = _env.step(np.random.randint(0, 2))
        reward_sum += reward
        if done:
            cur_episodes += 1
            print("Reward for this episode was:{}".format(reward_sum))
            reward_sum = 0
            _env.reset()
    _env.close()


def policy_gradient_run():

    obs_ph = tf.placeholder(tf.float32, shape=[None, _obs_dim], name='obs_input')
    act_ph = tf.placeholder(tf.int32, shape=[None], name='act_input')
    weights_ph = tf.placeholder(tf.float32, shape=[None], name='weights_input')

    # build policy gradient
    pg = PolicyGradient(n_feature=_obs_dim, n_action_space=_num_action_space, n_action_output=_num_action_output)
    #pg_logit = pg.mlp(obs_ph, sizes=[32] + [_num_action_space])
    pg.build_policy_network(obs_ph)
    pg_loss_op, pg_train_op = pg.train_policy_network(act_ph, weights_ph)
    pg_action = pg.action_policy_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(_epoch_num):
            # lists for storing observations, actions, rewards in a batch
            obs_batch_list = list()
            act_batch_list = list()
            reward_batch_list = list()
            weights_batch_list = list()

            # list for storing the number of actions in a batch
            act_len_list = list()

            # list for storing the reward in a episode
            reward_eps_list = list()

            # create the first obs for a episode
            obs = _env.reset()

            # Trajectories are also frequently called episodes or rollouts.
            # so, a trajectory is a episode here
            while True:
                obs_batch_list.append(obs)

                # reshape to fit the placeholder's shape [None, _obs_dim]
                act = sess.run(pg_action, feed_dict={obs_ph: obs.reshape(1, -1)})
                act_batch_list.append(act)

                # compute the next observation, reward, is_done according to the action
                obs, reward, done, info = _env.step(act)
                reward_eps_list.append(reward)

                if done:
                    eps_reward = sum(reward_eps_list)
                    eps_reward_len = len(reward_eps_list)
                    reward_batch_list.append(eps_reward)
                    act_len_list.append(eps_reward_len)

                    # the weight for each logprob(a|s) is R(tau)
                    # so each logprob should use the total reward for the
                    weights_batch_list += [eps_reward] * eps_reward_len

                    # reset episode-specific variables
                    obs, done, reward_eps_list = _env.reset(), False, []
                    
                    if len(obs_batch_list) > _batch_size:
                        break

            batch_loss, _ = sess.run([pg_loss_op, pg_train_op],
                                     feed_dict={
                                         obs_ph: np.array(obs_batch_list),
                                         act_ph: np.array(act_batch_list),
                                         weights_ph: np.array(weights_batch_list)
                                     })

            print("epoch:{}, batch loss:{}, batch reward:{}, actions in a eps:{}".format(i, batch_loss, np.mean(reward_batch_list), np.mean(act_len_list)))


if __name__ == "__main__":
    # how many epoch for training the policy network
    _epoch_num = 20

    # how many states (not episodes) in a batch
    _batch_size = 5000
    _num_action_output = 1
    _reward_decay = 0.99
    _training_threshold = 1000

    _env = gym.make('CartPole-v0')

    _num_action_space = _env.action_space.n
    _obs_dim = _env.observation_space.shape[0]

    # using policy random method to generate action
    # just for testing
    # test_random_action(100)

    policy_gradient_run()

    _env.close()
