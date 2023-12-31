import wandb
import yaml
from src.engine import Trainer
from src.utils import (
    set_seeds,
    save_checkpoint,
    get_used_params,
    find_best_run,
    get_sweep_variables,
    randomly_select_n_train_by_percentage,
)

with open("sweep_params2.yaml") as file:
    sweep_params = yaml.load(file, Loader=yaml.FullLoader)


def run_sweep(c: dict = None):
    global sweep_id

    run = wandb.init(config=c)
    c = wandb.config
    set_seeds(c.seed_no)

    graph_dataset, model, optimizer = get_sweep_variables(c)
    if c.n_train_percentage != 1:
        train_mask_new = randomly_select_n_train_by_percentage(
            graph_dataset.processed_dataset.train_mask, c.n_train_percentage
        )

        trainer = Trainer(model, optimizer, graph_dataset)
        trainer.train_mask = train_mask_new
    else:
        trainer = Trainer(model, optimizer, graph_dataset)

    trainer.pipeline(
        max_epochs=c.trainer_pipeline["max_epochs"],
        patience=c.trainer_pipeline["patience"],
        wandb_flag=True,
    )

    existing_best_test_acc, _ = find_best_run(target_dataset=c.dataset["dataset_name"])
    print("existing_best_test_acc: ", existing_best_test_acc)

    if existing_best_test_acc < trainer.best_test_acc:
        "We find new best model, saving it..."
        used_params = get_used_params(c)
        ckpt = dict(trainer=trainer, params=used_params)
        save_checkpoint(
            ckpt,
            c.dataset["dataset_name"],
            trainer.best_test_acc,
            sweep_id,
        )

    wandb.log(data={"test/best_test_acc": trainer.best_test_acc})
    wandb.log(data={"n_train_percentage_100": c.n_train_percentage * 100})


sweep_id = wandb.sweep(sweep_params, project="text-rgcn-percentage")
wandb.agent(sweep_id, project="text-rgcn-percentage", function=run_sweep)
