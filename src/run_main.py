"""
File that runs all experiments.
"""
from pathlib import Path

from hyper_parameters import KernelRegressionMode, ResNetMode
from main import run_experiment

"""
***************************************************************************************************
helper functions
***************************************************************************************************
"""

"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    Path("results").mkdir(exist_ok=True)
    # Run without KR and without skip connections
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
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
