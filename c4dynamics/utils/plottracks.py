# pragma: no cover
# import time
import argparse
import os
import sys

sys.path.append("")
sys.path.append("examples")
import pickle

# from enum import Enum
import numpy as np

# c4d.cprint('im in ' + os.getcwd(), 'y')
# for p in sys.path:
#   c4d.cprint(p, 'c')
from matplotlib import pyplot as plt

# from scipy.interpolate import interp1d
from programs._mtracks import Trkstate

import c4dynamics as c4d

# import multiprocessing
plt.style.use("dark_background")
plt.switch_backend("TkAgg")


def plottracks(fol, trkid=None, block=True):
    # print(fol)
    # print(trkid)
    # print(block)
    if trkid is not None:
        trkname = str(trkid)
    else:
        trkname = "all tracks"

    # def plottracks(fol, mtracks, pklpts, dt_video, dt, save_png, trkid = None):

    # load tracks, detections

    # the detections are simultenuous with the correct states.
    # but the detections are independent of the tracks and the tracks perhaps miss updates, then where should the detections be plotted?
    # but every thing is driven by the video. even if there are predicts in between. so
    #   where are these properties introduced here?

    # dtctsfile = os.path.join(fol, *[f for f in os.listdir(fol) if f.endswith('detections.pkl')]

    if os.path.exists(os.path.join(fol, "mtracks.pkl")):
        with open(os.path.join(fol, "mtracks.pkl"), "rb") as f:
            mtracks = pickle.load(f)
    else:
        raise ValueError("data folder doesnt consist tracks file")

    detections = {}

    if os.path.exists(os.path.join(fol, "detections.pkl")):
        # try subfol first:
        with open(os.path.join(fol, "detections.pkl"), "rb") as f:
            detections = pickle.load(f)
    elif os.path.exists(os.path.join(os.path.dirname(fol), "detections.pkl")):
        # then try parent folder:
        with open(os.path.join(os.path.dirname(fol), "detections.pkl"), "rb") as f:
            detections = pickle.load(f)
    else:
        c4d.cprint("data folder doesnt consist detections file")

    # if save_png == plotbackend.SAVE:
    #   plt.switch_backend('agg')
    # else:

    # X = []

    # for idx, p in enumerate(detections.values()):
    #   px = [idx * mtracks.dt_video] + p[0].X.tolist() if p else [idx * mtracks.dt_video] + [np.nan, np.nan, np.nan, np.nan]
    #   X.append(px)

    # X = np.array(X)
    # t2 = np.linspace(X[0, 0], X[-1, 0], 3 * len(X))
    # X1 = interp1d(X[:, 0], X[:, 1])(t2)
    # Y1 = interp1d(X[:, 0], X[:, 2])(t2)
    # W  = interp1d(X[:, 0], X[:, 3])(t2)
    # H  = interp1d(X[:, 0], X[:, 4])(t2)
    # VX1 = np.concatenate(([0], np.diff(X1) / t2[1]))
    # VY1 = np.concatenate(([0], np.diff(Y1) / t2[1]))

    # X = np.column_stack((t2, X1, Y1, W, H, VX1, VY1))

    trks = {**mtracks.trackers_hist, **mtracks.trackers}
    map = {
        "x": (0, 0),
        "y": (1, 0),
        "w": (0, 1),
        "h": (1, 1),  # , 'vx': (0, 2), 'vy': (1, 2)}
        "vx": (0, 0),
        "vy": (1, 0),
        "ax": (0, 1),
        "ay": (1, 1),
    }
    # color = [0.36601279, 0.46364415, 0.83173777] # np.random.rand(3)
    fig, axs = plt.subplots(2, 2, gridspec_kw={"wspace": 0.3, "hspace": 0.5})
    fig.suptitle(f"{trkname}", fontsize=16, fontweight="bold")
    fig2, axs2 = plt.subplots(2, 2, gridspec_kw={"wspace": 0.3, "hspace": 0.5})
    fig2.suptitle(f"{trkname}", fontsize=16, fontweight="bold")
    fig3, axs3 = plt.subplots(1, 2, gridspec_kw={"wspace": 0.3, "hspace": 0.5})
    fig3.suptitle(f"{trkname}", fontsize=16, fontweight="bold")
    fig4, axs4 = plt.subplots(1, 2, gridspec_kw={"wspace": 0.3, "hspace": 0.5})
    fig4.suptitle(f"{trkname}", fontsize=16, fontweight="bold")

    #
    # draw raw detections
    ##
    for idx, p in detections.items():
        t = idx * mtracks.dt_video
        px = np.array([pi.X for pi in p if p])
        if len(px) == 0:
            continue

        for j, var in enumerate(["x", "y", "w", "h"]):
            axs[map[var]].plot(
                t * np.ones(px.shape[0]),
                px[:, j],
                linewidth=0,
                color=np.array([1, 1, 1]),
                marker="o",
                markersize=3,
                markerfacecolor="none",
                label=var + " raw",
            )

            if var == "x" or var == "y":
                print(map[var][0])
                axs3[map[var][0]].plot(
                    t * np.ones(px.shape[0]),
                    px[:, j],
                    linewidth=0,
                    color=np.array([1, 1, 1]),
                    marker="o",
                    markersize=4,
                    markerfacecolor="none",
                    label=var + " raw",
                )

    #
    # draw tracks and uncertaties
    ##
    np.random.seed(0)
    color = np.array([1.0, 0.0, 1.0])  # magenta
    color = np.array([0.0, 1.0, 1.0])  # cyan
    i = 0

    for k, v in trks.items():
        if trkid is not None and all(f != k for f in trkid):
            continue
        print(k)

        # if trkid is None:
        #   color = np.random.rand(3)
        # elif all(f != k for f in trkid):
        #   # trkid is not none and k is not in trkid
        #   continue

        if len(v.data()) == 0:
            continue

        # fig, axs = plt.subplots(2, 2, gridspec_kw = {'wspace': 0.2, 'hspace': 0.5})
        iscorrect = np.where(
            np.vectorize(lambda x: x.value)(v.data("state")[1]) == Trkstate.CORRECTED.value
        )[0]

        for var in ["x", "y", "w", "h"]:

            x = v.data(var)[1]
            t_sig, x_sig = eval('v.data("p' + var + '")')

            axs[map[var]].plot(t_sig, x, linewidth=2, color=np.array(v.color) / 255, label=var)
            # ±std
            axs[map[var]].plot(
                t_sig, x + np.sqrt(x_sig.squeeze()), linewidth=1, color=color, label="std"
            )  # np.array(v.color) / 255)
            axs[map[var]].plot(
                t_sig, x - np.sqrt(x_sig.squeeze()), linewidth=1, color=color
            )  # np.array(v.color) / 255)
            # correct
            axs[map[var]].plot(
                t_sig[iscorrect],
                x[iscorrect],
                linewidth=0,
                color=color,
                marker="o",
                markersize=2,
                markerfacecolor="none",
                label="correct",
            )

            if var == "x" or var == "y":
                axs3[map[var][0]].plot(
                    t_sig, x, linewidth=4, color=np.array(v.color) / 255, label=var
                )
                # ±std
                axs3[map[var][0]].plot(
                    t_sig, x + np.sqrt(x_sig.squeeze()), linewidth=2, color=color, label="std"
                )  # np.array(v.color) / 255)
                axs3[map[var][0]].plot(
                    t_sig, x - np.sqrt(x_sig.squeeze()), linewidth=2, color=color
                )  # np.array(v.color) / 255)
                # correct
                axs3[map[var][0]].plot(
                    t_sig[iscorrect],
                    x[iscorrect],
                    linewidth=0,
                    color=color,
                    marker="o",
                    markersize=2,
                    markerfacecolor="none",
                    label="correct",
                )

            if i == 0:

                if hasattr(v, "ax"):
                    f2attrs = ["vx", "vy", "ax", "ay"]
                else:
                    f2attrs = ["vx", "vy"]

                if var == "y":
                    axs[map[var]].invert_yaxis()
                    axs3[map[var][0]].invert_yaxis()

                c4d.plotdefaults(axs[map[var]], var, "t", "")

                if var == "x" or var == "y":
                    c4d.plotdefaults(axs3[map[var][0]], var, "t", "")

        for var in f2attrs:

            x = v.data(var)[1]
            t_sig, x_sig = eval('v.data("p' + var + '")')

            axs2[map[var]].plot(t_sig, x, linewidth=2, color=np.array(v.color) / 255, label=var)
            # ±std
            axs2[map[var]].plot(
                t_sig, x + np.sqrt(x_sig.squeeze()), linewidth=1, color=color, label="std"
            )  # np.array(v.color) / 255)
            axs2[map[var]].plot(
                t_sig, x - np.sqrt(x_sig.squeeze()), linewidth=1, color=color
            )  # np.array(v.color) / 255)
            # correct
            axs2[map[var]].plot(
                t_sig[iscorrect],
                x[iscorrect],
                linewidth=0,
                color=color,
                marker="o",
                markersize=2,
                markerfacecolor="none",
                label="correct",
            )

            if var == "vx" or var == "vy":
                axs4[map[var][0]].plot(
                    t_sig, x, linewidth=4, color=np.array(v.color) / 255, label=var
                )
                # ±std
                axs4[map[var][0]].plot(
                    t_sig, x + np.sqrt(x_sig.squeeze()), linewidth=2, color=color, label="std"
                )  # np.array(v.color) / 255)
                axs4[map[var][0]].plot(
                    t_sig, x - np.sqrt(x_sig.squeeze()), linewidth=2, color=color
                )  # np.array(v.color) / 255)
                # correct
                axs4[map[var][0]].plot(
                    t_sig[iscorrect],
                    x[iscorrect],
                    linewidth=0,
                    color=color,
                    marker="o",
                    markersize=2,
                    markerfacecolor="none",
                    label="correct",
                )

            if i == 0:
                c4d.plotdefaults(axs2[map[var]], var, "t", "")

                if var == "vx" or var == "vy":
                    c4d.plotdefaults(axs4[map[var][0]], var, "t", "")

        color = np.random.rand(3)
        i += 1

    fig.savefig(os.path.join(fol, trkname + ".png"))
    fig2.savefig(os.path.join(fol, trkname + "_2.png"))
    fig3.savefig(os.path.join(fol, trkname + "_3.png"))
    fig4.savefig(os.path.join(fol, trkname + "_4.png"))

    # if save_png != plotbackend.SAVE:
    if block:
        plt.show(block=True)


if __name__ == "__main__":

    # if len(sys.argv) == 1:
    #   sys.argv.append('--fol')
    #   sys.argv.append('examples\out\cars1')

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument(
        "--vidname", required=True, help="folder name containing the pickle to explore"
    )
    parser.add_argument("--trkid", nargs="+", type=int, help="trk id to filter")
    # store_true means if provided without following param then set it to true.
    parser.add_argument(
        "--debug", action="store_true", default=False, help="whether to run in debug mode"
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    # Convert parsed arguments to a dictionary
    args_dict = vars(args)
    # Print the dictionary
    c4d.cprint(args_dict, "y")

    if args.debug:
        input(
            "Run python debugger using process id. \nSelect the pyenv process. \nPress to continue and wait"
        )
    args_dict.pop("debug")

    videoname = args.vidname
    args_dict.pop("vidname")
    pickledir = os.path.join("examples", "_out", videoname)

    plottracks(pickledir, **args_dict)
    # plot_process = multiprocessing.Process(target = plottracks, args = (args_dict, ))
    # plot_process.start()

    # input('continue running with background plot..')

    # for a in sys.argv:
    #   c4d.cprint(a, 'm')
    # input('press to cont')
    # sys.argv.append('examples\out\cars1')
    # sys.argv.append('69')

    # if len(sys.argv) < 2:
    #   raise AttributeError('folder name is missing')
    # elif len(sys.argv) == 2:
    #   fol = sys.argv[1]
    #   trkid = None
    # elif len(sys.argv) == 3:
    #   fol = sys.argv[1]
    #   trkid = int(sys.argv[2])
    # else:
    #   fol = sys.argv[1]
    #   trkid = int(sys.argv[2])
    #   if any(l == 'debug' for l in sys.argv):
    #     # try:
    #     input('run python debugger using process id. select the pyenv process. press to continue and wait')
    # except Exception as e:
    # time.sleep(2)
    # print(f'error: {e}')

    # fol = 'examples\out\cars1'
    # ptspath = os.path.join(fol, 'detections.pkl')
