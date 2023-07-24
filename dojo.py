from __future__ import absolute_import, division, print_function
import base64
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
import datetime


class Dojo:
    def __init__(
        self,
        q_network,
        env,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
        td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss,
        log_steps=10,
        log_dir="tensorboard",
        training_batch_size=64,
    ):
        self.train_step_counter = tf.Variable(0)
        self.environment = env
        self.agent = tf_agents.agents.dqn.dqn_agent.DqnAgent(
            self.environment.time_step_spec(),
            self.environment.action_spec(),
            q_network=q_network,
            optimizer=optimizer,
            td_errors_loss_fn=td_errors_loss_fn,
            train_step_counter=self.train_step_counter,
        )

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.environment.batch_size,
            max_length=100000,
        )

        # Setup training parameters
        self._log_steps = log_steps
        self._training_batch_size = training_batch_size

        # Setup tensorboard metrics
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
        self._train_return = tf.keras.metrics.Mean("train_return", dtype=tf.float32)
        self._train_summary_writer = tf.summary.create_file_writer(
            log_dir + "/" + current_time + "/train"
        )

    # Computes the average return of the current agent
    def _avg_return(self, num_episodes=10):
        total_return = 0.0

        for _ in range(num_episodes):
            time_step = self.environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def _collect_step(self):
        time_step = self.environment.current_time_step()
        action_step = self.agent.collect_policy.action(time_step)
        next_time_step = self.environment.step(action_step.action)
        traj = tf_agents.trajectories.trajectory.from_transition(
            time_step, action_step, next_time_step
        )

        self.replay_buffer.add_batch(traj)

    def train(self, iterations):
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self._training_batch_size,
            num_steps=2,
        ).prefetch(3)
        iterator = iter(dataset)
        self.environment.reset()

        # Collect enough data to be able to get a batch from training
        for _ in range(self._training_batch_size):
            self._collect_step()

        # Run iteration steps of collection and training
        for _ in range(iterations):
            self._collect_step()

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience).loss

            # Update the metrics
            self._train_loss(train_loss)

            # Log metrics periodically
            step = self.train_step_counter.numpy()
            if step % self._log_steps == 0:
                # Compute the average return over 10 runs
                avg_return = self._avg_return()
                self._train_return(avg_return)

                # Log metrics to tensorboard
                with self._train_summary_writer.as_default():
                    tf.summary.scalar("loss", self._train_loss.result(), step=step)
                    tf.summary.scalar("return", self._train_return.result(), step=step)
                print(
                    "step = {0}: loss = {1} return = {2}".format(
                        step, self._train_loss.result(), self._train_return.result()
                    )
                )

                self._train_loss.reset_states()
                self._train_return.reset_states()
