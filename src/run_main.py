"""
File that runs all experiments.
"""
import pathlib
import shutil
from pathlib import Path

from hyper_parameters import KernelRegressionMode, ResNetMode
from main import run_experiment

"""
***************************************************************************************************
helper functions
***************************************************************************************************
"""

def remove_dir(name):
    path = Path(name)
    if path.exists():
        shutil.rmtree(path=path)

"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    remove_dir("lightning_logs")
    remove_dir("results")
    Path("results").mkdir()
    # Run without KR and without skip connections
    depths = list(range(1, 20))
    # accuracies = []
    # losses = []
    for depth in depths:
        d = run_experiment(
            seed=42,
            depth=depth,
            use_kr=KernelRegressionMode.OFF,
            res_net_mode=ResNetMode.NONE
        )
        # test_accuracy = d["test accuracy"]
        # test_loss = d["test loss"]
        import json
        with open(f'results/exp1_{depth}.json', 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)

        # accuracies += [test_accuracy]
        # losses += [test_loss]
        # print(accuracies)
        # print(losses)


"""
***************************************************************************************************
run main
***************************************************************************************************
"""

if __name__ == "__main__":
    main()
