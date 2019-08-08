#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

from utils import str2bool
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import mlflow
from gym import wrappers
from time import time


def main():
    with mlflow.start_run(run_name="Gen Expert Data"):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('expert_policy_file', type=str)
        parser.add_argument('envname', type=str)
        parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--max_timesteps", type=int)
        parser.add_argument('--num_rollouts', type=int, default=20,
                            help='Number of expert roll outs')
        args = parser.parse_args()
        for k, v in vars(args).items():
            mlflow.log_param(k, v)

        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        print('loaded and built')

        with tf.Session():
            tf_util.initialize()
            returns = []
            observations = []
            actions = []
            steps = []
            for i in range(args.num_rollouts):
                env = gym.make(args.envname)
                if args.render:
                    video_dir = "./videos/{0}/".format(time())
                    env = wrappers.Monitor(env, video_dir, force=True)
                max_steps = args.max_timesteps or env.spec.max_episode_steps
                print("max_steps set to {0}".format(max_steps))
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                trial_steps = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    #action = env.action_space.sample()
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    trial_steps += 1
                    if args.render:
                        env.render(mode='rgb_array')
                    if trial_steps % 100 == 0: print("%i/%i"%(trial_steps, max_steps))
                    if trial_steps >= max_steps:
                        print("hit max_steps")
                        break
                returns.append(totalr)
                steps.append(trial_steps)
                env.close()
                if args.render:
                    mlflow.log_artifacts(video_dir)

            for s in steps:
                mlflow.log_metric('steps', s)
            for r in returns:
                mlflow.log_metric('returns', r)
            mlflow.log_metric('mean return', np.mean(returns))
            mlflow.log_metric('std of return', np.std(returns))

            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}

            filename = os.path.join('expert_data', args.envname + '.pkl')
            with open((filename), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
            mlflow.log_artifact(filename, artifact_path="expert_data_file")


if __name__ == '__main__':
    main()
