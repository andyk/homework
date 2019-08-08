# make a TF NN and train it on the observations and actions
# that we observed when we rolled out our expert policies.

import argparse
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow.tensorflow
import numpy as np

mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument('expert_run_id', type=str)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()
for k, v in vars(args).items():
    mlflow.log_param(k, v)

# ===================================================
# read in dataset from expert policy rollouts.
mlflow_c = mlflow.tracking.MlflowClient()
expert_data_file_base = mlflow_c.download_artifacts(args.expert_run_id, "")
expert_data_file_rel_path = mlflow_c.list_artifacts(args.expert_run_id, "expert_data_file")[0].path
expert_data_file_full_path = expert_data_file_base + "/" + expert_data_file_rel_path
print("opening {0}".format(expert_data_file_full_path))

with open(expert_data_file_full_path, 'rb') as f:
    data = pickle.loads(f.read())
    expert_observations = data['observations']
    expert_actions = data['actions']
    # Reshape actions
    expert_actions = expert_actions.reshape((expert_actions.shape[0], expert_actions.shape[2]))


# ===================================================
# normalize & standardize the observations
print("mean: {0}, std: {1}".format(expert_observations.mean(), expert_observations.std()))
mlflow.log_params({"norm_mean": expert_observations.mean(), "norm_std": expert_observations.std()})
print(expert_observations[0])
def norm(x):
    return (x - x.mean()) / x.std()
expert_observations = norm(expert_observations)
print("----")
print(expert_observations[0])

print(expert_observations.shape)
print(type(expert_observations.shape))
print(expert_actions.shape)


# ====================================================
# Build the model
# Try convolutional, recurrent layers.
model = tf.keras.Sequential([
    keras.layers.Dense(128, input_shape=expert_observations[0].shape, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(expert_actions.shape[1])
])

model.compile(optimizer="adam",
              loss="mse",
              metrics=["mae", "mse"])

# ====================================================
# Train / test split
obs_train, obs_test, actions_train, actions_test = train_test_split(expert_observations,
                                                                    expert_actions,
                                                                    test_size=0.1)

# ====================================================
# fit the model
print("expert_observations shape: {0}".format(expert_observations.shape))
print("expert_actions shape: {0}".format(expert_actions.shape))
print("--")
print("obs_train shape: {0}".format(obs_train.shape))
print("actions_train shape: {0}".format(actions_train.shape))
print("obs_test shape: {0}".format(obs_test.shape))
print("actions_test shape: {0}".format(actions_test.shape))
model.fit(obs_train, actions_train, batch_size=args.batch_size, epochs=args.num_epochs)

# ====================================================
# Evaluate the model

# copied from https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

predicted_test_actions = model.predict(obs_test)
print("predicted_test_actions.shape: {0}".format(predicted_test_actions.shape))
rmse, mae, r2 = eval_metrics(actions_test, predicted_test_actions)
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

score = model.evaluate(obs_test, actions_test)
print("score: {0}:".format(score))
