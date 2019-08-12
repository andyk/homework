# Copying Holly Grimm's solution https://github.com/hollygrimm/cs294-homework/blob/master/hw1/bc.py
# Copy and pasting and merging it into a copy of my behavior_cloner.py code.

import argparse
import pickle
import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow.tensorflow
import gym
from gym import wrappers
from tqdm import tqdm

#Imports copied from hollygrimm's solution
import logging
from hollygrimm_model import Model

# The following doesn't seem to work with the way Holly Grimm builds her tensorflow model.
mlflow.tensorflow.autolog()

def config_logging(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def create_model(session, obs_samples, num_observations, num_actions, logger, optimizer,
                 learning_rate, restore, checkpoint_dir):
    model = Model(obs_samples, num_observations, num_actions, checkpoint_dir, logger,
                  optimizer, learning_rate)

    if restore:
        model.load(session)
    else:
        logger.info("Created model with fresh parameters")
        session.run(tf.global_variables_initializer())

    return model


def bc(expert_data_filename, env_name, restore, results_dir, max_timesteps=None,
       optimizer='adam', num_epochs=100, learning_rate=.001, batch_size=32, keep_prob=1):
    # Reset TF env
    tf.reset_default_graph()

    # Create a gym env.
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.max_episode_steps

    with open(expert_data_filename, 'rb') as f:
        data = pickle.loads(f.read())

    obs = np.stack(data['observations'], axis=0)
    actions = np.squeeze(np.stack(data['actions'], axis=0))

    x_train, x_test, y_train, y_test = train_test_split(obs, actions, test_size=0.2)

    num_samples = len(x_train)

    min_val_loss = sys.maxsize

    with tf.Session() as session:
        model = create_model(session, x_train, x_train.shape[1], y_train.shape[1], logger,
                             optimizer, learning_rate, restore, results_dir)

        file_writer = tf.summary.FileWriter(results_dir, session.graph)
        #file_writer = tf.summary.FileWriter(results_dir, session.graph)

        for epoch in tqdm(range(num_epochs)):
            perm = np.random.permutation(x_train.shape[0])

            obs_samples = x_train[perm]
            action_samples = y_train[perm]

            loss = 0.
            for k in range(0, obs_samples.shape[0], batch_size):
                batch_loss, training_scalar = model.update(session, obs_samples[k:k + batch_size],
                                                           action_samples[k:k + batch_size],
                                                           keep_prob)
                loss += batch_loss

            file_writer.add_summary(training_scalar, epoch)

            min_val_loss, validation_scalar = validate(model, logger, session, x_test, y_test,
                                                       epoch, batch_size, min_val_loss, results_dir)
            file_writer.add_summary(validation_scalar, epoch)

            # Test the updated model after each epoch of training the DNN.
            new_exp = model.test_run(session, env, max_steps)
            tqdm.write(
                "Epoch %3d; Loss %f; Reward %f; Steps %d" % (epoch, loss / num_samples,
                                                             new_exp['reward'], new_exp['steps']))

        # Write a video of the final gym test results.
        env = wrappers.Monitor(env, results_dir, force=True)

        results = []
        for _ in tqdm(range(10)):
            results.append(model.test_run(session, env, max_steps)['reward'])
        logger.info("Reward mean and std dev with behavior cloning: %f(%f)" % (np.mean(results),
                                                                               np.std(results)))
        mlflow.log_params({"reward_mean": np.mean(results), "reward_std": np.std(results)})
    return np.mean(results), np.std(results)


def validate(model, logger, session, x_test, y_test, num_epoch, batch_size, min_loss, checkpoint_dir):
    avg_loss = []

    # for k in range(0, x_test.shape[0], batch_size):
    loss, validation_scalar = model.validate(session, x_test, y_test)
    avg_loss.append(loss)

    new_loss = sum(avg_loss) / len(avg_loss)
    logger.info("Finished epoch %d, average validation loss = %f" % (num_epoch, new_loss))

    if new_loss < min_loss:  # Only save model if val loss dropped
        model.save(session)
        min_loss = new_loss
    return min_loss, validation_scalar


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('expert_run_id', type=str)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--restore", type=bool, default=False)
    args = parser.parse_args()
    for k, v in vars(args).items():
        mlflow.log_param(k, v)

    if not os.path.exists('results'):
        os.makedirs('results')
    log_file = os.path.join(os.getcwd(), 'results', 'train_out.log')
    logger = config_logging(log_file)

    #env_models = [('Ant-v1', 'data/Ant-v1_data_250_rollouts.pkl', 'experts/Ant-v1.pkl', 250),
    #              ('HalfCheetah-v1', 'data/HalfCheetah-v1_data_10_rollouts.pkl', 'experts/HalfCheetah-v1.pkl', 10),
    #              ('Hopper-v1', 'data/Hopper-v1_data_10_rollouts.pkl', 'experts/Hopper-v1.pkl',  10),
    #              ('Humanoid-v1', 'data/Humanoid-v1_data_250_rollouts.pkl', 'experts/Humanoid-v1.pkl', 250),
    #              ('Reacher-v1', 'data/Reacher-v1_data_250_rollouts.pkl', 'experts/Reacher-v1.pkl', 250),
    #              ('Walker2d-v1', 'data/Walker2d-v1_data_10_rollouts.pkl','experts/Walker2d-v1.pkl', 10)
    #              ]

    #for env_name, rollout_data, expert_policy_file, num_rollouts in env_models :

    # ===================================================
    # read in dataset from expert policy rollouts.
    mlflow_c = mlflow.tracking.MlflowClient()
    expert_data_file_base = mlflow_c.download_artifacts(args.expert_run_id, "")
    expert_data_file_rel_path = mlflow_c.list_artifacts(args.expert_run_id, "expert_data_file")[
        0].path
    expert_data_filename = expert_data_file_base + "/" + expert_data_file_rel_path
    print("opening {0}".format(expert_data_filename))

    env_name = mlflow_c.get_run(args.expert_run_id).data.params["envname"]

    bc_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'bc')
    bc_reward_mean, bc_reward_std = bc(expert_data_filename, env_name, args.restore, bc_results_dir,
                                       batch_size=args.batch_size, num_epochs=args.num_epochs)

    logger.info('Behavior Cloning mean & std rewards: %f(%f))' %
                (bc_reward_mean, bc_reward_std))
    print("logging 'results' directory to mlflow.")
    mlflow.log_artifacts('results')

    # Commenting out dagger for now.
    #da_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'da')
    #if not os.path.exists(da_results_dir):
    #    os.makedirs(da_results_dir)
    #_,_, da_mean,da_std = dagger(rollout_data, expert_policy_file, env_name, args.restore, da_results_dir, num_rollouts)
    #results.append((env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std))

    #for env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std in results :
    #    logger.info('Env: %s, Expert: %f(%f), Behavior Cloning: %f(%f), Dagger: %f(%f)'%
    #          (env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std))

