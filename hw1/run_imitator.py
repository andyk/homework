import argparse
import gym
from gym import wrappers
import mlflow
from utils import str2bool
import numpy as np
from time import time

with mlflow.start_run(run_name="Imitator Series"):
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('behavior_clone_training_run_id', type=str)
    parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--model_type', type=str, default="keras")
    args = parser.parse_args()
    for k, v in vars(args).items():
        mlflow.log_param(k, v)

    mlflow_model_uri = "runs:/" + args.behavior_clone_training_run_id + "/model"
    if args.model_type.lower() == "keras":
        import mlflow.keras
        print("loading cloned keras behavior model from mlflow")
        model = mlflow.keras.load_model(mlflow_model_uri)
    elif args.model_type.lower() == "tf" or args.model_type.lower() == "tensorflow":
        import mlflow.tensorflow
        print("loading cloned tensorflow behavior model from mlflow")
        model = mlflow.tensorflow.load_model(mlflow_model_uri)


    returns = []

    mlflow_c = mlflow.tracking.MlflowClient()
    run = mlflow_c.get_run(args.behavior_clone_training_run_id)
    norm_mean, norm_std = run.data.params["norm_mean"], run.data.params["norm_std"]
    def norm(x):
        return (x - norm_mean) / norm_std

    for i in range(args.num_rollouts):
        with mlflow.start_run(run_name="Imitator Trial", nested=True):
            env = gym.make(args.envname)
            if args.render:
                video_dir = "./videos/{0}/".format(time())
                env = wrappers.Monitor(env, video_dir, force=True)
            max_steps = args.max_timesteps or env.spec.max_episode_steps
            obs = env.reset()
            done = False
            i_r = 0.
            steps = 0
            actions = []
            while not done:
                #action = model.predict(norm(obs.reshape(1, 376,)))
                action = env.action_space.sample()
                actions.append(action)
                obs, r, done, _ = env.step(action)
                i_r += r
                steps += 1
                if args.render:
                    env.render(mode='rgb_array')
                if steps > max_steps:
                    break
            returns.append(i_r)
            actions_np = np.array(actions)
            print("Trial {0} complete. Reward:{1}, Steps: {2}".format(i, i_r, steps))
            mlflow.log_metrics({"Reward": i_r, "Steps": steps})
            env.close()
            if args.render:
                mlflow.log_artifacts(video_dir)

    print("All {0} rollouts complete. Avg Reward: {1}".format(args.num_rollouts,
                                                            np.array(returns).mean()))
    mlflow.log_metrics({"Num Trials": args.num_rollouts, "Avg Reward": np.array(returns).mean()})
