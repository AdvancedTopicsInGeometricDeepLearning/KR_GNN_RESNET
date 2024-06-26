"""
File that runs all experiments.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

"""
***************************************************************************************************
helper functions
***************************************************************************************************
"""


def save_plot(title: str, add_title=True):
    if add_title:
        plt.title(title)
    save_to_file = True
    if save_to_file:
        dir_name = "graphs"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(f"{dir_name}/{title}.png", bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def get_xy_from_experiment(exp: int, y_tag: str, x_tag: str):
    path_list = Path(f"./results/exp{exp}").glob("*.json")

    xy = []
    for path in path_list:
        with open(path) as json_file:
            data = json.load(json_file)
            x = data[x_tag]
            y = data[y_tag]
            xy += [(x, y)]
    xy.sort(key=lambda tup: tup[0])

    x = [tup[0] for tup in xy]
    y = [tup[1] for tup in xy]
    return x, y


def make_exp_plot(
        experiments: list[tuple[int, str]], y_tag: str, x_tag: str, title: str,
        x_title: str, y_title: str
):
    list_of_xy = []
    for exp, _ in experiments:
        xy = get_xy_from_experiment(exp=exp, x_tag=x_tag, y_tag=y_tag)
        list_of_xy += [xy]

    # plot
    fig, ax = plt.subplots()
    for (x, y), (_, t) in zip(list_of_xy, experiments):
        ax.plot(x, y, label=t)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.legend()
    save_plot(title=title)


"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    experiments = [(1, "no KR + no ResNet"), (2, "KR + no ResNet"), (3, "KR + ResNet 2")]
    make_exp_plot(
        experiments=experiments,
        y_tag="train loss", x_tag="depth",
        title="Training loss per depth",
        x_title="depth", y_title="training loss"
    )
    make_exp_plot(
        experiments=experiments,
        y_tag="train accuracy", x_tag="depth",
        title="Training accuracy per depth",
        x_title="depth", y_title="training accuracy"
    )
    make_exp_plot(
        experiments=experiments,
        y_tag="val loss", x_tag="depth",
        title="Validation loss per depth",
        x_title="depth", y_title="validation loss"
    )
    make_exp_plot(
        experiments=experiments,
        y_tag="val accuracy", x_tag="depth",
        title="Validation accuracy per depth",
        x_title="depth", y_title="validation accuracy"
    )
    make_exp_plot(
        experiments=experiments,
        y_tag="test loss", x_tag="depth",
        title="Testing loss per depth",
        x_title="depth", y_title="testing loss"
    )
    make_exp_plot(
        experiments=experiments,
        y_tag="test accuracy", x_tag="depth",
        title="Testing accuracy per depth",
        x_title="depth", y_title="testing accuracy"
    )


"""
***************************************************************************************************
run main
***************************************************************************************************
"""

if __name__ == "__main__":
    main()
