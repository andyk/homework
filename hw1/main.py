"""
Copied & modified https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/main.py
Solution for Berkeley Deep RL course homework 1: https://github.com/berkeleydeeprlcourse/homework/tree/master/hw1
Requires mujoco pro 150 to be installed and have valid license.
"""

import click
import os


import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.status != RunStatus.FINISHED:
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run_info.run_id, run_info.status))
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(("Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)") % previous_version, git_commit)
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

# Params from the three entry points
#   generate_expert_data
#       expert_policy_file: string
#       envname: string
#
#   train_behavior_clone:
#       expert_run_id: string
#
#   run_imitator:
#       envname: string
#       behavior_clone_training_run_id: string
#       render: {type: string, default: false}


@click.command()
@click.option("--envname", default="Humanoid-v2", type=str)
@click.option("--expert_policy_file", default="experts/Humanoid-v2.pkl", type=str)
@click.option("--render_expert", default="False", type=str)
@click.option("--render_bc", default="True", type=str)
@click.option("--num_expert_rollouts", default=5, type=int)
@click.option("--num_bc_trials", default=1, type=int)
def workflow(envname, expert_policy_file, render_expert, render_bc, num_expert_rollouts, num_bc_trials):
    # Note: The entrypoint names are defined in MLproject.
    with mlflow.start_run(run_name="HW1 Main Entry Point") as active_run:
        os.environ['SPARK_CONF_DIR'] = os.path.abspath('.')
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        generate_expert_data_run = _get_or_run("generate_expert_data",
                                               {"expert_policy_file": expert_policy_file,
                                                "envname": envname,
                                                "num_rollouts": num_expert_rollouts,
                                                "render": render_expert},
                                               git_commit)

        train_behavior_clone_run = _get_or_run("train_behavior_clone",
                                               {"expert_run_id": generate_expert_data_run.info.run_id},
                                               git_commit)

        run_imitator_run = _get_or_run("run_imitator",
                                       {"behavior_clone_training_run_id": train_behavior_clone_run.info.run_id,
                                        "envname": envname,
                                        "num_rollouts": num_bc_trials,
                                        "render": render_bc},
                                       git_commit)

if __name__ == '__main__':
    workflow()