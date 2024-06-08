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


def make_exp_plot(exp: int, y_tag: str, x_tag: str, title: str):
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

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    save_plot(title=title)


"""
***************************************************************************************************
main function
***************************************************************************************************
"""


def main():
    make_exp_plot(
        exp=1, y_tag="train loss", x_tag="depth",
        title="Training loss per depth (no KR, no ResNet)"
    )
    make_exp_plot(
        exp=1, y_tag="train accuracy", x_tag="depth",
        title="Training accuracy per depth (no KR, no ResNet)"
    )


"""
***************************************************************************************************
run main
***************************************************************************************************
"""

if __name__ == "__main__":
    main()
