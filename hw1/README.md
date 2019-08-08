# Andy Konwinski's solution to CS294-112 HW 2: Imitation Learning

This directory is an MLflow project. It is a multi-step workflow, which
means that there are 4 entry points defined in in the MLProject file.

You should be able to use reproduce my results on your laptop using
MLflow Project feature. To run all 3 stages of the homework solution
on your laptop do the following:

1) make sure you have installed git and conda (used for reproducibility in MLflow)
2) install MLflow: `pip install mlflow`
3) use MLflow Run to reproduce my results: `mlflow run git@github.com:andyk/homework.git#hw1`

Note that running this project without any entry point specified defaults to the
"main" entry point which:
    (a) Generates expert data that we will use to do behavior cloning.
        Runs 5 rollouts using OpenAI Gym with Humanoid-v2 env and the expert
        policy (a trained TensorFlow neural net) provided by the Berkeley CS294-112 
        staff (in the hw1 `experts` dir). See `run_expert.py`.
    (b) Performs behavior cloning by training a deep net.
        Trains a 4 layer Keras deep neural network (all Dense layers), but
        first standardizes the data (mean center, normalize by std)
    (c) Runs 5 new Rollouts of the same gym env, this time though uses our
        newly trained deep net to make decision about which action to take
        based on the observation it receives from env at each step.

See `main.py` to see the MLflow worklflow defined.

You can also run the individual stages of the multi-step workflow manually one
at a time by calling mlflow run with each of the three entry points in succession
with appropriate parameters (the 2nd two stages take as input the run_id of the
previous stage so that they can download the output files from the previous
stage), e.g.:
* `mlflow run git@github.com:andyk/homework.git#hw1 -e generate_expert_data ...`
* `mlflow run git@github.com:andyk/homework.git#hw1 -e train_behavior_clone ...`
* `mlflow run git@github.com:andyk/homework.git#hw1 -e run_imitator ...`


I've updated OpenAI gym dependency from 0.10.5 to 0.14.0 since I was
having problems with render() in 0.10.5.

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.14.0**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.
