import numpy as np
from IPython import embed
import collections
import os

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.utils import common
from tf_agents.trajectories.time_step import TimeStep
from tensorflow.python.framework.tensor_spec import TensorSpec, BoundedTensorSpec

IMG_WIDTH = 320
IMG_HEIGHT = 180
TASK_OBS_DIM = 4


class SACAgent:
    def __init__(
            self,
            root_dir,
            conv_1d_layer_params=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            conv_2d_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)],
            encoder_fc_layers=[256],
            actor_fc_layers=[256],
            critic_obs_fc_layers=[256],
            critic_action_fc_layers=[256],
            critic_joint_fc_layers=[256],
            # Params for target update
            target_update_tau=0.005,
            target_update_period=1,
            # Params for train
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            alpha_learning_rate=3e-4,
            td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
            gamma=0.99,
            reward_scale_factor=1.0,
            gradient_clipping=None,
            # Params for eval
            eval_deterministic=False,
            # Params for summaries and logging
            debug_summaries=False,
            summarize_grads_and_vars=False
    ):
        '''A simple train and eval for SAC.'''

        root_dir = os.path.expanduser(root_dir)
        policy_dir = os.path.join(root_dir, 'train', 'policy')

        time_step_spec = TimeStep(
            TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
            TensorSpec(shape=(), dtype=tf.float32, name='reward'),
            BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount',
                              minimum=np.array(0., dtype=np.float32), maximum=np.array(1., dtype=np.float32)),
            collections.OrderedDict({
                'task_obs': BoundedTensorSpec(shape=(TASK_OBS_DIM,), dtype=tf.float32, name=None,
                                            minimum=np.array(-3.4028235e+38, dtype=np.float32),
                                            maximum=np.array(3.4028235e+38, dtype=np.float32)),
                'depth': BoundedTensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32, name=None,
                                           minimum=np.array(-1.0, dtype=np.float32),
                                           maximum=np.array(1.0, dtype=np.float32)),
                'rgb': BoundedTensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32, name=None,
                                         minimum=np.array(-1.0, dtype=np.float32),
                                         maximum=np.array(1.0, dtype=np.float32)),
            })
        )
        observation_spec = time_step_spec.observation
        action_spec = BoundedTensorSpec(shape=(2,), dtype=tf.float32, name=None,
                                        minimum=np.array(-1.0, dtype=np.float32),
                                        maximum=np.array(1.0, dtype=np.float32))

        glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        preprocessing_layers = {}
        if 'rgb' in observation_spec:
            preprocessing_layers['rgb'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'depth' in observation_spec:
            preprocessing_layers['depth'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'task_obs' in observation_spec:
            preprocessing_layers['task_obs'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if len(preprocessing_layers) <= 1:
            preprocessing_combiner = None
        else:
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=actor_fc_layers,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
            kernel_initializer=glorot_uniform_initializer,
        )

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()
        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)
        tf_agent.initialize()

        if eval_deterministic:
            self.eval_py_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
        else:
            self.eval_py_policy = tf_agent.policy

        policy_checkpointer = common.Checkpointer(
            ckpt_dir=policy_dir,
            policy=tf_agent.policy,
            global_step=global_step)

        policy_checkpointer.initialize_or_restore()

        # activate the session
        obs = {
            'depth': np.ones((IMG_HEIGHT, IMG_WIDTH, 1)),
            'rgb': np.ones((IMG_HEIGHT, IMG_WIDTH, 3)),
            'task_obs': np.ones((TASK_OBS_DIM,))
        }
        action = self.act(obs)
        print('activate TF session')
        print('action', action)

    def reset():
        pass

    def act(self, obs):
        batch_obs = {}
        for key in obs:
            batch_obs[key] = np.expand_dims(obs[key], axis=0)
        time_step = TimeStep(
            np.ones(1),
            np.ones(1),
            np.ones(1),
            batch_obs,
        )
        policy_state = ()

        action_step = self.eval_py_policy.action(time_step, policy_state)
        action = action_step.action[0]
        return action


if __name__ == "__main__":
    obs = {
        'depth': np.ones((IMG_HEIGHT, IMG_WIDTH, 1)),
        'rgb': np.ones((IMG_HEIGHT, IMG_WIDTH, 3)),
        'task_obs': np.ones((TASK_OBS_DIM,))
    }
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    agent = SACAgent(root_dir='test')
    action = agent.act(obs)
    print('action', action)
