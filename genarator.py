# %%
import torch
from src.utils import find_best_run
from src.engine import Trainer

# Finding best checkpoint
for dataset_name in ["ohsumed", "R8", "R52", "mr"]:
    ckpt_folder_path, ckpt_file_path = find_best_run(target_dataset=dataset_name)
    ckpt = torch.load(ckpt_file_path)

    trainer: Trainer = ckpt["trainer"]

    trainer.model = trainer.best_model
    _, test_acc = trainer.eval_model()
    print("Test Acc: ", test_acc)
    if test_acc == trainer.best_test_acc:
        print("Test Passed for dataset: ", dataset_name)
    else:
        print("Test Failed for dataset: ", dataset_name)

# %%
