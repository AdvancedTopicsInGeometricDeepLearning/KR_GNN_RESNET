"""
File that runs all experiments.
"""
import gc
import shutil
from multiprocessing import Process
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


def run_once(depth, test: int):
    d = {}
    match test:
        case 1:
            d = run_experiment(
                seed=42,
                depth=depth,
                use_kr=KernelRegressionMode.OFF,
                res_net_mode=ResNetMode.NONE
            )
        case 2:
            d = run_experiment(
                seed=42,
                depth=depth,
                use_kr=KernelRegressionMode.AFTER_EACH_BLOCK,
                res_net_mode=ResNetMode.NONE
            )
        case 3:
            d = run_experiment(
                seed=42,
                depth=depth,
                use_kr=KernelRegressionMode.AFTER_EACH_BLOCK,
                res_net_mode=ResNetMode.ADD
            )
        case _:
            assert False
    # test_accuracy = d["test accuracy"]
    # test_loss = d["test loss"]
    import json
    with open(f'results/exp{test}/{depth}.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    remove_dir("lightning_logs")
    # Run without KR and without skip connections
    depths = list(range(1, 30))
    exp = 2
    remove_dir(f"results/exp{exp}")
    Path(f"results/exp{exp}").mkdir(parents=True)
    # accuracies = []
    # losses = []
    for depth in depths:
        p = Process(target=run_once, args=[depth, exp])
        p.start()
        p.join()


"""
***************************************************************************************************
run main
***************************************************************************************************
"""

if __name__ == "__main__":
    main()
