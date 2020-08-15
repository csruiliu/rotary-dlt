import tensorflow as tf
from tf_agents.utils import common
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer


class MLSchEngine:
    def __init__(self, sch_env):
        self.mlsch_env = sch_env
        self.rl_agent = None
        self.replay_buffer = None
        self.reward_list = list()

    def build_sch_agent(self):
        actor_net = actor_distribution_network.ActorDistributionNetwork(self.mlsch_env.observation_spec(),
                                                                        self.mlsch_env.action_spec(),
                                                                        fc_layer_params=(100,))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step_counter = tf.Variable(0)

        self.rl_agent = reinforce_agent.ReinforceAgent(self.mlsch_env.time_step_spec(), self.mlsch_env.action_spec(),
                                                       actor_network=actor_net, optimizer=optimizer,
                                                       normalize_returns=True, train_step_counter=train_step_counter)
        self.rl_agent.initialize()
        self.rl_agent.train = common.function(self.rl_agent.train)
        self.rl_agent.train_step_counter.assign(0)

    def train_sch_agent(self, num_train_iterations, collect_episodes_per_iteration,
                        steps_num_per_batch, log_interval, eval_interval, num_eval_episodes_for_train):
        self._init_replay_buffer(steps_num_per_batch)

        for i in range(num_train_iterations):
            print('training iteration {0}'.format(i))
            # Collect a few step using collect_policy and save to the replay buffer.
            self._replay_collect_episode(collect_episodes_per_iteration)

            experience = self.replay_buffer.gather_all()
            train_loss = self.rl_agent.train(experience)
            self.replay_buffer.clear()

            step = self.rl_agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % eval_interval == 0:
                avg_reward = self.evaluate_sch_agent(num_eval_episodes_for_train)
                print('evaluate the agent at step = {0}: Average Return = {1}'.format(step, avg_reward))
                self.reward_list.append(avg_reward)

        return self.reward_list

    def evaluate_sch_agent(self, eval_num_episodes):
        print('evaluate the agent for {} episodes.'.format(eval_num_episodes))
        total_return = 0.0
        for _ in range(eval_num_episodes):
            time_step = self.mlsch_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.rl_agent.policy.action(time_step)
                time_step = self.mlsch_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_reward = total_return / eval_num_episodes
        return avg_reward

    def benchmark_before_training(self, benchmark_num_episodes=10):
        # Evaluate the agent's policy once before training.
        avg_return = self.evaluate_sch_agent(benchmark_num_episodes)
        self.reward_list.append(avg_return)
        print('Returns before training:{}'.format(self.reward_list))

    def _init_replay_buffer(self, steps_num_per_batch):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.rl_agent.collect_data_spec,
                                                                            batch_size=self.mlsch_env.batch_size(),
                                                                            max_length=steps_num_per_batch)

    def _replay_collect_episode(self, collect_num_episodes):
        episode_counter = 0

        while episode_counter < collect_num_episodes:
            current_time_step = self.mlsch_env.current_time_step()
            action_step = self.rl_agent.collect_policy.action(current_time_step)
            next_time_step = self.mlsch_env.step(action_step.action)
            traj = trajectory.from_transition(current_time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            if self.replay_buffer is None:
                raise TypeError('replay buffer has not been initialize')

            self.replay_buffer.add_batch(traj)

            if traj.is_boundary():
                self.mlsch_env.reset()
                episode_counter += 1

    def _replay_collect_step(self, environment, policy, num_steps):
        step_counter = 0

        while step_counter < num_steps:
            current_time_step = environment.current_time_step()
            action_step = policy.action(current_time_step)
            next_time_step = environment.simulated_step(action_step.action)
            traj = trajectory.from_transition(current_time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            self.replay_buffer.add_batch(traj)

            step_counter += 1

    def reset(self):
        self.rl_agent = None
        self.replay_buffer = None
        self.reward_list.clear()
