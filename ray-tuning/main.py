import click
import os
import yaml
import pickle
from deepcorr.preprocess import Preprocess
from deepcorr.training import Server

import ray
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler


class HyperParamOptExp(ray.tune.Trainable):
    def setup(self, config):
        self.config = config
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids()))

        for i in self.config["hyperopt_utils"]:
            if "layers" in i["name"]:
                if "conv" in i["name"]:
                    layers_idx = self.config["conv_layers_choice"]
                else:
                    layers_idx = self.config["dense_layers_choice"]

                self.config[i["name"]] = i["values"][layers_idx]
            else:
                self.config[i["name"]] = i["values"][self.config["conv_layers_choice"]]

        self.model = Server(config=self.config)

    def step(self):

        self.report = self.model.train_epoch(self.training_iteration)

        return {**self.report}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, str(self.trial_id) + "_model")
        self.model.save_model(checkpoint_path)
        pickle.dump(
            self.report, open(os.path.join(checkpoint_dir, "results.pkl"), "wb")
        )
        return checkpoint_path
        pass

    def load_checkpoint(self, checkpoint_path):
        self.model.load_model(checkpoint_path)


################################################################################
# Settings
################################################################################
@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--preprocess", type=bool, default=False, help="Preprocess data.")
@click.option("--gpu", type=click.Choice(['0','1']), help="Use GPU for training.")
def main(config_path, preprocess, gpu):

    # Connect to ray instance
    ray.init(address="auto")

    # Load config file and set path as working dir
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    os, os.chdir(config_dir)

    with open(config_path) as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)

    if preprocess:
        p = Preprocess(config_yml=config_path)
        p.main_run()

    # Make paths absolute
    for k in exp_config["data"]:
        if "path" in k:
            try:
                exp_config["data"][k] = os.path.abspath(exp_config["data"][k])
            except:
                pass

    # Load exp settings
    OPTIM_METRIC = exp_config["exp"]["optim_metric"]
    EXP_NAME = exp_config["exp"]["exp_name"]
    NUM_REPEAT = exp_config["exp"]["num_repeat"]
    GRACE_PERIOD = exp_config["exp"]["sched_grace_period"]
    EPOCHS = exp_config["exp"]["epochs"]
    NUM_SAMPLES = exp_config["exp"]["num_samples"]
    EXP_PATH = exp_config["exp"]["exp_path"]

    # Initialize ax for search space definition
    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(
        name="f{EXP_NAME}Search",
        parameters=exp_config["hyperopt"],
        objective_name=OPTIM_METRIC,
    )

    search_alg = AxSearch(ax_client=ax)
    re_search_alg = Repeater(search_alg, repeat=NUM_REPEAT)

    sched = ASHAScheduler(
        time_attr="training_iteration",
        grace_period=GRACE_PERIOD,
        mode="max",
        metric=OPTIM_METRIC,
    )

    analysis = ray.tune.run(
        HyperParamOptExp,
        name=EXP_NAME,
        resume="AUTO",
        stop={
            "training_iteration": EPOCHS,
        },
        resources_per_trial={"cpu": 1, "gpu": int(gpu)},
        num_samples=NUM_SAMPLES,
        local_dir=EXP_PATH,
        search_alg=re_search_alg,
        scheduler=sched,
        config=exp_config,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        checkpoint_score_attr=OPTIM_METRIC,
    )

if __name__ == "__main__":
    main()
