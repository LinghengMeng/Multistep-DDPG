import tensorflow as tf
import numpy as np
import pandas as pd
import pybulletgym
import gym
import time
from mddpg.core import MLP
from mddpg.utils.logx import EpochLogger
import os.path as osp
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs1_sim_state_buf = []
        self.obs1_ela_steps = np.zeros(size, dtype=np.int)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_sim_state_buf = []
        self.obs2_ela_steps = np.zeros(size, dtype=np.int)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, obs_sim_state, obs_elapsed_steps, act, rew,
              next_obs, next_obs_sim_state, next_obs_elapsed_steps, done):
        self.obs1_buf[self.ptr] = obs
        self.obs1_sim_state_buf.append(obs_sim_state)
        self.obs1_ela_steps[self.ptr] = obs_elapsed_steps
        self.obs2_buf[self.ptr] = next_obs
        self.obs2_sim_state_buf.append(next_obs_sim_state)
        self.obs2_ela_steps[self.ptr] = next_obs_elapsed_steps
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_batch_n_step(self, batch_size=32, n_step=1, debug=False):
        """
        return training batch for n-step experiences
        :param batch_size:
        :param n_step:
        :return: dict:
            'obs1': batch_size x n_step x obs_dim
            'obs2': batch_size x n_step x obs_dim
            'acts': batch_size x n_step x act_dim
            'rews': batch_size x n_step
            'done': batch_size x n_step
        """
        idxs = np.random.randint(0, self.size - (n_step - 1), size=batch_size)
        batch_obs1 = np.zeros([batch_size, n_step, self.obs_dim])
        batch_obs2 = np.zeros([batch_size, n_step, self.obs_dim])
        batch_obs1_sim_state = []
        batch_obs2_sim_state = []
        batch_obs1_ela_steps = np.zeros([batch_size, n_step])
        batch_obs2_ela_steps = np.zeros([batch_size, n_step])
        batch_acts = np.zeros([batch_size, n_step, self.act_dim])
        batch_rews = np.zeros([batch_size, n_step])
        batch_rews_potential = np.zeros([batch_size, n_step])
        batch_done = np.zeros([batch_size, n_step])
        for i in range(n_step):
            batch_obs1[:, i, :] = self.obs1_buf[idxs + i]
            batch_obs2[:, i, :] = self.obs2_buf[idxs + i]
            batch_obs1_ela_steps[:, i] = self.obs1_ela_steps[idxs + i]
            batch_obs2_ela_steps[:, i] = self.obs2_ela_steps[idxs + i]
            batch_acts[:, i, :] = self.acts_buf[idxs + i]
            batch_rews[:, i] = self.rews_buf[idxs + i]
            batch_done[:, i] = self.done_buf[idxs + i]

        # Simulation state corresponds to obs2 for restoring
        for i in idxs:
            batch_obs1_sim_state.append([self.obs1_sim_state_buf[i + s_i] for s_i in range(n_step)])
            batch_obs2_sim_state.append([self.obs2_sim_state_buf[i + s_i] for s_i in range(n_step)])

        # Set all done after the fist met one to 1
        done_index = np.asarray(np.where(batch_done == 1))
        for d_i in range(done_index.shape[1]):
            x, y = done_index[:, d_i]
            batch_done[x, y:] = 1
        if debug:
            import pdb;
            pdb.set_trace()

        batch_done = np.hstack((np.zeros((batch_size, 1)), batch_done))
        return dict(obs1=batch_obs1[:, 0, :],
                    obs2=batch_obs2[:, :, :],
                    obs1_sim_state=batch_obs1_sim_state,
                    obs2_sim_state=batch_obs2_sim_state,
                    obs1_ela_steps=batch_obs1_ela_steps,
                    obs2_ela_steps=batch_obs2_ela_steps,
                    acts=batch_acts[:, 0, :],
                    rews=batch_rews,
                    rews_potential=batch_rews_potential,
                    done=batch_done)


def pybulletenv_get_state(env):
    """
    Function used to get state information from PyBullet physics engine.
    :param env:
    :return:
    """
    body_num = env.env._p.getNumBodies()
    # body_info = [env.env._p.getBodyInfo(body_i) for body_i in range(body_num)]
    floor_id, robot_id = 0, 1

    robot_base_pos_ori = env.env._p.getBasePositionAndOrientation(robot_id)
    robot_base_vel = env.env._p.getBaseVelocity(robot_id)

    joint_num = env.env._p.getNumJoints(robot_id)
    joint_state = []
    for joint_i in range(joint_num):
        joint_state.append(env.env._p.getJointState(robot_id, joint_i))

    state = {'body_num': body_num,
             'robot_base_pos_ori': robot_base_pos_ori, 'robot_base_vel': robot_base_vel,
             'joint_num': joint_num, 'joint_state': joint_state}
    return state


def pybulletenv_set_state(env, state):
    """
    Function used to set state in PyBullet physics engine.
    :param env:
    :param state:
    :return:
    """

    body_num = env.env._p.getNumBodies()
    floor_id, robot_id = 0, 1
    joint_num = env.env._p.getNumJoints(robot_id)
    if body_num != state['body_num'] and joint_num != state['body_num']:
        print('Set state error.')
    # restore state
    env.env._p.resetBasePositionAndOrientation(robot_id,
                                               state['robot_base_pos_ori'][0],
                                               state['robot_base_pos_ori'][1])
    env.env._p.resetBaseVelocity(robot_id, state['robot_base_vel'][0], state['robot_base_vel'][1])
    for j_i, j_s in enumerate(state['joint_state']):
        env.env._p.resetJointState(robot_id, j_i, j_s[0], j_s[1])
    return env


def get_sim_state_and_elapsed_steps(env, env_name):
    if 'PyBulletEnv' in env_name or 'MuJoCoEnv' in env_name:
        # o_sim_state = env.env._p.saveState()
        o_sim_state = pybulletenv_get_state(env)
    else:
        o_sim_state = env.sim.get_state()  # MuJuco
    o_elapsed_steps = env._elapsed_steps
    return o_sim_state, o_elapsed_steps


def restore_simulation_state(env, env_name, res_sim_state, res_obs, res_ela_steps):
    """
    Function used to restore simulation state.
    """
    # restore simulator state
    _ = env.reset()  # (crucial to reset env._elapsed_steps)
    env._elapsed_steps = res_ela_steps  # Crucial to reset elapsed step in simulator
    if 'PyBulletEnv' in env_name or 'MuJoCoEnv' in env_name:
        # env.robot.feet_contact is calculated in env.step(), so we need to reset it after restore joints and position.
        # https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/roboschool/envs/locomotion/walker_base_env.py
        # 1. restoreState
        # 2. restore feet_contact
        # 3. call env.robot.calc_state() to prepare walk_target_dist for env.robot.calc_potential()
        # 4. restore potential (used to calculate progress but only updated when calling env.step())
        env = pybulletenv_set_state(env, res_sim_state)
        env.robot.feet_contact = res_obs[-len(env.robot.feet):].copy()
        # Crucial call sequence: 1. calc_state() 2. calc_potential()
        #   (as calc_potential() needs walk_target_dist which is calculated in calc_state() which further depends on state.)
        obs_after_restore = env.robot.calc_state()
        env.env.potential = env.robot.calc_potential()
    else:
        # Different from PyBulletGym, forward_reward is calculated by state in env.step() rather than keeping
        #   potential proterty.
        # 1. restore state
        # 2. call env.sim.forward() to complete restoration state
        env.sim.set_state(res_sim_state)  # MuJuco
        env.sim.forward()
        obs_after_restore = env.env._get_obs()
        # state_after_restore = env.sim.get_state()

    # # examine if restore is correct.
    # allowed_obs_err = 1e-4
    # if (abs(res_obs - obs_after_restore) > allowed_obs_err).any():
    #     import pdb;
    #     pdb.set_trace()
    #     raise Exception('Restore state fail in online_expand_first_n_step!')

    return env


def online_expand_to_end(sess, q, pi, x_ph, a_ph, gamma,
                         env_name, env,
                         replay_buffer, n_step, exp_batch_size=10, exp_n_step=0):
    """
    This function is used to expand policy on environment to get ground true value for Q.
    """
    exp_batch = replay_buffer.sample_batch_n_step(exp_batch_size, n_step=n_step)

    # restore to state after exp_n_step step, then expand until termination.
    restore_obs2_sim = pd.DataFrame(exp_batch['obs2_sim_state']).iloc[:, 0].tolist()
    exp_outs = []
    for exp_b_i in range(exp_batch_size):
        # restore simulation state
        res_sim_state = restore_obs2_sim[exp_b_i]
        res_obs = exp_batch['obs2'][exp_b_i, 0, :]
        res_ela_steps = exp_batch['obs2_ela_steps'][exp_b_i, 0].copy()
        env = restore_simulation_state(env, env_name, res_sim_state, res_obs, res_ela_steps)

        # expand until termination
        exp_o2 = exp_batch['obs2'][exp_b_i, exp_n_step, :].copy()  # np.copy() to copy array, otherwise copy reference
        exp_done = exp_batch['done'][exp_b_i, exp_n_step + 1].copy()
        exp_dis_r = 0
        step = 0
        while not exp_done:
            exp_o2, exp_r, exp_done, _ = env.step(
                sess.run(pi, feed_dict={x_ph: exp_o2.reshape(1, -1)})[0])
            exp_dis_r += (gamma ** step) * exp_r
            step += 1
        exp_outs.append(exp_dis_r)

    # Online expansion
    # expand based on current policy until termination, after first exp_n_step
    dis_acc_rew = np.asarray(exp_outs)
    # exp_n_step from replay buffer
    dis_r_n_step = np.tile(np.asarray([gamma ** i for i in range(exp_n_step + 1)]).reshape([1, -1]),
                           [exp_batch_size, 1])
    dis_acc_n_step = np.sum(
        np.multiply(dis_r_n_step, exp_batch['rews'][:, :exp_n_step + 1].reshape([exp_batch_size, -1])),
        axis=1)
    ground_truth_q = dis_acc_n_step + gamma ** (exp_n_step + 1) * dis_acc_rew
    # Prediction
    predicted_q = sess.run(q, feed_dict={x_ph: exp_batch['obs1'], a_ph: exp_batch['acts']})
    return ground_truth_q, predicted_q


def online_expand_first_n_step(sess, pi, backups, dis_acc_first, dis_acc_following, dis_boots,
                               x_ph, x2_ph, r_ph, d_ph, batch_size_ph,
                               env_name, env,
                               replay_buffer, n_step, exp_batch_size=10):
    # Sample mini-batch
    exp_batch = replay_buffer.sample_batch_n_step(exp_batch_size, n_step=n_step)
    # create matrix to record n-step experiences where the first step is restored from past experience
    exp_rew = np.zeros(exp_batch['rews'].shape)
    exp_next_obs = np.zeros(exp_batch['obs2'].shape)
    exp_done = np.zeros(exp_batch['done'].shape)
    exp_next_obs[:, 0, :] = exp_batch['obs2'][:, 0, :].copy()
    exp_rew[:, 0] = exp_batch['rews'][:, 0].copy()
    exp_done[:, 0:2] = exp_batch['done'][:, 0:2].copy()

    # Prepare restore states
    restore_obs2_sim = pd.DataFrame(exp_batch['obs2_sim_state']).iloc[:, 0].tolist()
    for exp_b_i in range(exp_batch_size):
        # restore simulation state
        res_sim_state = restore_obs2_sim[exp_b_i]
        res_obs = exp_next_obs[exp_b_i, 0, :]
        res_ela_steps = exp_batch['obs2_ela_steps'][exp_b_i, 0].copy()
        env = restore_simulation_state(env, env_name, res_sim_state, res_obs, res_ela_steps)

        # expand n-1 steps following 1st step
        o2 = exp_next_obs[exp_b_i, 0, :].copy()
        done = exp_done[exp_b_i, 1].copy()  # the 1st done is only used to calculate accumulated reward

        for s_i in range(n_step - 1):  # the following n-1 steps
            if done:
                break
            else:
                o2, r, done, _ = env.step(sess.run(pi, feed_dict={x_ph: o2.reshape(1, -1)})[0])
                exp_next_obs[exp_b_i, 1 + s_i, :] = o2
                exp_rew[exp_b_i, 1 + s_i] = r
                exp_done[exp_b_i, 2 + s_i] = done

    done_index = np.asarray(np.where(exp_done == 1))
    for d_i in range(done_index.shape[1]):
        x, y = done_index[:, d_i]
        exp_done[x, y:] = 1

    # 1st step + (n-1) step online expansion + bootstrapping
    exp_backup, exp_first, exp_second, exp_third = sess.run([backups[n_step - 1], dis_acc_first[n_step - 1],
                                                             dis_acc_following[n_step - 1], dis_boots[n_step - 1]],
                                                            feed_dict={x2_ph: exp_next_obs,
                                                                       r_ph: exp_rew,
                                                                       d_ph: exp_done,
                                                                       batch_size_ph: exp_batch_size})
    # 1st step + (n-1) step offline expansion + bootstrapping
    n_s_backup, n_s_first, n_s_second, n_s_third = sess.run([backups[n_step - 1], dis_acc_first[n_step - 1],
                                                             dis_acc_following[n_step - 1], dis_boots[n_step - 1]],
                                                            feed_dict={x2_ph: exp_batch['obs2'],
                                                                       r_ph: exp_batch['rews'],
                                                                       d_ph: exp_batch['done'],
                                                                       batch_size_ph: exp_batch_size})
    # if (exp_second > 200).any():
    #     import pdb;
    #     pdb.set_trace()
    return exp_backup, exp_first, exp_second, exp_third, \
           n_s_backup, n_s_first, n_s_second, n_s_third


"""

Deep Deterministic Policy Gradient (DDPG)

"""


def ddpg_n_step_new(env_name, render_env=False,
                    actor_hidden_layers=[300, 300], critic_hidden_layers=[300, 300],
                    seed=0,
                    steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
                    n_step=1, backup_method='mixed_n_step', exp_batch_size=10,
                    without_delay_train=False,
                    log_n_step_offline_and_online_expansion=False,
                    log_n_step_online_expansion_and_boostrapping=False,
                    polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
                    act_noise=0.1, policy_delay=2, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Inputs to computation graph
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
    a_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))
    x2_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, obs_dim))
    r_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    d_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    n_step_ph = tf.placeholder(dtype=tf.float32, shape=())
    batch_size_ph = tf.placeholder(dtype=tf.int32)

    actor_hidden_sizes = actor_hidden_layers
    critic_hidden_sizes = critic_hidden_layers
    actor_hidden_activation = tf.keras.activations.relu
    actor_output_activation = tf.keras.activations.tanh
    critic_hidden_activation = tf.keras.activations.relu
    critic_output_activation = tf.keras.activations.linear

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        actor = MLP(layer_sizes=actor_hidden_sizes + [act_dim],
                    hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        critic = MLP(layer_sizes=critic_hidden_sizes + [1],
                     hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)
        pi = act_limit * actor(x_ph)
        q = tf.squeeze(critic(tf.concat([x_ph, a_ph], axis=-1)), axis=1)
        q_pi = tf.squeeze(critic(tf.concat([x_ph, pi], axis=-1)), axis=1)

    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        actor_targ = MLP(layer_sizes=actor_hidden_sizes + [act_dim],
                         hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        critic_targ = MLP(layer_sizes=critic_hidden_sizes + [1],
                          hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)

    n_step_bootstrapped_q = []
    for n_step_i in range(n_step):
        # slice next_obs for different n
        next_obs_tmp = tf.reshape(tf.slice(x2_ph, [0, n_step_i, 0], [batch_size_ph, 1, obs_dim]),
                                  [batch_size_ph, obs_dim])
        pi_targ_tmp = act_limit * actor_targ(next_obs_tmp)
        q_pi_targ_tmp = tf.squeeze(critic_targ(tf.concat([next_obs_tmp, pi_targ_tmp], axis=-1)), axis=1)
        n_step_bootstrapped_q.append(q_pi_targ_tmp)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Bellman backup for Q function
    dis_acc_first, dis_acc_following, dis_boots, n_step_backups = [], [], [], []
    for n_step_i in range(1, n_step + 1):
        # for k = 0,..., n-1: (1-done) * gamma**(k) * reward
        dis_rate = tf.tile(tf.reshape(tf.pow(gamma, tf.range(0, n_step_i, dtype=tf.float32)), [1, -1]),
                           [batch_size_ph, 1])
        dis_rate = tf.multiply(dis_rate, 1 - tf.slice(d_ph, [0, 0], [batch_size_ph, n_step_i]))  # multiply done slice
        n_step_dis_rew = tf.multiply(dis_rate, tf.slice(r_ph, [0, 0], [batch_size_ph, n_step_i]))
        # first step reward
        n_step_first_rew = n_step_dis_rew[:, 0]
        # discounted following step reward
        n_step_following_rew = n_step_dis_rew[:, 1:]
        n_step_offline_acc_rew = tf.reduce_sum(n_step_following_rew, axis=1)
        # discounted bootstrapped reward
        boots_q = gamma ** n_step_i * (1 - tf.reshape(tf.slice(d_ph, [0, n_step_i], [batch_size_ph, 1]), [-1])) * \
                  n_step_bootstrapped_q[n_step_i - 1]
        # whole n-step backup
        backup_tmp = tf.stop_gradient(n_step_first_rew + n_step_offline_acc_rew + boots_q)
        # Separately save for logging
        dis_acc_first.append(n_step_first_rew), dis_acc_following.append(n_step_offline_acc_rew)
        dis_boots.append(boots_q), n_step_backups.append(backup_tmp)

    # Define different backup methods
    backup_avg_n_step = tf.stop_gradient(tf.reduce_mean(tf.stack(n_step_backups, axis=1), axis=1))
    backup_min_n_step = tf.stop_gradient(tf.reduce_min(tf.stack(n_step_backups, axis=1), axis=1))
    backup_avg_n_step_exclude_1 = tf.stop_gradient(tf.reduce_mean(tf.stack(n_step_backups[1:], axis=1), axis=1))
    backups = [backup_avg_n_step, backup_min_n_step, backup_avg_n_step_exclude_1] + n_step_backups

    # Crucial: if statement here does not work in tensorflow
    backup_flag = np.zeros((3 + int(n_step)))
    if backup_method == 'avg_n_step':
        backup_flag[0] = 1
    elif backup_method == 'min_n_step':
        backup_flag[1] = 1
    elif backup_method == 'avg_n_step_exclude_1':
        backup_flag[2] = 1
    else:
        tmp_step, _ = backup_method.split('_')
        tmp_step = int(tmp_step)
        if 1 <= tmp_step and tmp_step <= n_step:
            backup_flag[3 + tmp_step - 1] = 1  # index start from 0
        else:
            raise Exception('Wrong backup_method!')

    if np.sum(backup_flag) != 1:
        raise Exception('Wrong backup_flag!')

    backup_index = np.where(backup_flag == 1)[0][0]
    backup = backups[backup_index]
    print("backup_index={}".format(backup_index))

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q - backup) ** 2)

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    train_pi_op = pi_optimizer.minimize(loss=pi_loss, var_list=actor.variables)

    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_q_op = q_optimizer.minimize(loss=q_loss, var_list=critic.variables)

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                              for v_main, v_targ in zip(actor.variables + critic.variables,
                                                        actor_targ.variables + critic_targ.variables)])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(actor.variables + critic.variables,
                                                      actor_targ.variables + critic_targ.variables)])

    # Initialize variables and target networks
    sess = tf.keras.backend.get_session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # restore actor-critic model
    restore_actor_critic_model = False
    if restore_actor_critic_model:
        model_path = r"C:\Users\Lingheng\Google Drive\git_repos\spinup_data\2020-01-12_ddpg_n_step_new_AntPyBulletEnv_v0\2020-01-12_10-15-53-ddpg_n_step_new_AntPyBulletEnv_v0_s0"

        actor.load_weights(os.path.join(model_path, 'checkpoints', 'epoch90_actor'))
        critic.load_weights(os.path.join(model_path, 'checkpoints', 'epoch90_critic'))
        sess.run(target_init)

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    # For PyBulletGym  envs, must call env.render() before env.reset().
    if render_env:
        env.render()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o_sim_state, o_elapsed_steps = get_sim_state_and_elapsed_steps(env, env_name)
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        if render_env:
            env.render()
        o2, r, d, _ = env.step(a)
        o2_sim_state, o2_elapsed_steps = get_sim_state_and_elapsed_steps(env, env_name)

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, o_sim_state, o_elapsed_steps, a, r,
                            o2, o2_sim_state, o2_elapsed_steps, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o, o_sim_state, o_elapsed_steps = o2, o2_sim_state, o2_elapsed_steps

        if without_delay_train:
            # batch = replay_buffer.sample_batch(batch_size)
            batch = replay_buffer.sample_batch_n_step(batch_size, n_step=n_step)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done'],
                         n_step_ph: n_step,
                         batch_size_ph: batch_size}

            # Q-learning update
            outs = sess.run([q_loss, q, train_q_op], feed_dict)
            logger.store(LossQ=outs[0], QVals=outs[1])

            # # Policy update
            # if t % policy_delay == 0:
            # Delayed policy update
            outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
            logger.store(LossPi=outs[0])

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """

            if not without_delay_train:
                for j in range(ep_len):
                    # batch = replay_buffer.sample_batch(batch_size)
                    batch = replay_buffer.sample_batch_n_step(batch_size, n_step=n_step)
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done'],
                                 n_step_ph: n_step,
                                 batch_size_ph: batch_size}
                    # critic update
                    outs = sess.run([q_loss, q, backups, train_q_op], feed_dict)
                    logger.store(LossQ=outs[0], QVals=outs[1])
                    logger.store(QBackupAvgNStep=outs[2][0], QBackupMinNStep=outs[2][1],
                                 QBackupAvgNStepExclude1=outs[2][2])
                    logger.store(
                        **{'QBackup{}Step'.format(n_step_i + 1): outs[2][3 + n_step_i] for n_step_i in range(n_step)})
                    # actor update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])
            """
            ###############################################
            """
            if log_n_step_offline_and_online_expansion:
                # Logging: n-step offline + online expansion
                # start_time_1 = time.time()
                exp_after_n_step = 0  # indicates expand based on online policy after observing exp_n_step offline experiences
                ground_truth_q, predicted_q = online_expand_to_end(sess, q, pi, x_ph, a_ph, gamma,
                                                                   env_name, env,
                                                                   replay_buffer, n_step, exp_batch_size,
                                                                   exp_after_n_step)
                logger.store(PredictedQ=predicted_q, GroundTruthQ=ground_truth_q)
                # end_time_1 = time.time()
            if log_n_step_online_expansion_and_boostrapping:
                # Logging: n-step online expansion + bootstrapped Q thereafter
                # start_time_2 = time.time()
                exp_backup, exp_first, exp_second, exp_third, \
                n_s_backup, n_s_first, n_s_second, n_s_third = online_expand_first_n_step(sess, pi,
                                                                                          backups, dis_acc_first,
                                                                                          dis_acc_following, dis_boots,
                                                                                          x_ph, x2_ph, r_ph, d_ph,
                                                                                          batch_size_ph,
                                                                                          env_name, env,
                                                                                          replay_buffer, n_step,
                                                                                          exp_batch_size)

                logger.store(NStepOfflineBackup=n_s_backup, NStepOfflineFir=n_s_first,
                             NStepOfflineSec=n_s_second, NStepOfflineThi=n_s_third,
                             NStepOnlineBackup=exp_backup, NStepOnlineFir=exp_first,
                             NStepOnlineSec=exp_second, NStepOnlineThi=exp_third)
                # end_time_2 = time.time()
                # print('Method 1: {}, Method 2:{}'.format(end_time_1 - start_time_1, end_time_2-start_time_2))
            """
            ###############################################
            """

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            o_sim_state, o_elapsed_steps = get_sim_state_and_elapsed_steps(env, env_name)

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save actor-critic model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                model_save_dir = os.path.join(logger.output_dir, 'checkpoints')
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                actor.save_weights(os.path.join(model_save_dir, 'epoch{}_actor'.format(epoch)))
                critic.save_weights(os.path.join(model_save_dir, 'epoch{}_critic'.format(epoch)))

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            for n_step_i in range(n_step):
                logger.log_tabular('QBackup{}Step'.format(n_step_i + 1), with_min_and_max=True)
            logger.log_tabular('QBackupAvgNStep', with_min_and_max=True)
            logger.log_tabular('QBackupMinNStep', with_min_and_max=True)
            logger.log_tabular('QBackupAvgNStepExclude1', with_min_and_max=True)
            if log_n_step_offline_and_online_expansion:
                logger.log_tabular('PredictedQ', with_min_and_max=True)
                logger.log_tabular('GroundTruthQ', with_min_and_max=True)
            if log_n_step_online_expansion_and_boostrapping:
                logger.log_tabular('NStepOfflineBackup', with_min_and_max=True)
                logger.log_tabular('NStepOnlineBackup', with_min_and_max=True)
                logger.log_tabular('NStepOfflineFir', with_min_and_max=True)
                logger.log_tabular('NStepOnlineFir', with_min_and_max=True)
                logger.log_tabular('NStepOfflineSec', with_min_and_max=True)
                logger.log_tabular('NStepOnlineSec', with_min_and_max=True)
                logger.log_tabular('NStepOfflineThi', with_min_and_max=True)
                logger.log_tabular('NStepOnlineThi', with_min_and_max=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetahPyBulletEnv-v0')
    parser.add_argument('--render_env', action="store_true")
    parser.add_argument('--actor_hidden_layers', nargs='+', type=int, default=[300, 300])
    parser.add_argument('--critic_hidden_layers', nargs='+', type=int, default=[300, 300])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--n_step', type=int, default=5)
    parser.add_argument('--backup_method', type=str,
                        choices=['avg_n_step', 'min_n_step', 'avg_n_step_exclude_1',
                                 '1_step', '2_step', '3_step', '4_step', '5_step',
                                 '6_step', '7_step', '8_step', '9_step', '10_step'],
                        default='avg_n_step')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--without_delay_train', action='store_true')

    parser.add_argument('--log_n_step_offline_and_online_expansion', action='store_true')
    parser.add_argument('--log_n_step_online_expansion_and_boostrapping', action='store_true')
    parser.add_argument('--exp_batch_size', type=int, default=100, help='batch size for logging expansion of policy')

    parser.add_argument('--exp_name', type=str, default='mddpg')
    parser.add_argument('--act_noise', type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default='mddpg_data')

    args = parser.parse_args()

    # Set log data saving directory
    from mddpg.utils.run_utils import setup_logger_kwargs

    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        args.data_dir)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    ddpg_n_step_new(env_name=args.env, render_env=args.render_env,
                    actor_hidden_layers=args.actor_hidden_layers,
                    critic_hidden_layers=args.critic_hidden_layers,
                    act_noise=args.act_noise,
                    gamma=args.gamma, seed=args.seed, replay_size=args.replay_size,
                    n_step=args.n_step, backup_method=args.backup_method,
                    exp_batch_size=args.exp_batch_size,
                    without_delay_train=args.without_delay_train,
                    log_n_step_offline_and_online_expansion=args.log_n_step_offline_and_online_expansion,
                    log_n_step_online_expansion_and_boostrapping=args.log_n_step_online_expansion_and_boostrapping,
                    epochs=args.epochs, save_freq=args.save_freq,
                    batch_size=args.batch_size,
                    steps_per_epoch=args.steps_per_epoch, start_steps=args.start_steps,
                    logger_kwargs=logger_kwargs)
