# pragma: no cover

import os
import sys

sys.path.append("")
import pickle

import c4dynamics as c4d

if __name__ == "__main__":

    for a in sys.argv:
        c4d.cprint(a, "m")

    if len(sys.argv) >= 2:
        fol = sys.argv[1]
    else:
        raise AttributeError("folder name is missing")

    # fol = 'examples\out\cars1'
    ptspath = os.path.join(fol, "detections.pkl")

    if os.path.exists(ptspath):
        print("path exists")
        with open(ptspath, "rb") as file:
            pklpts = pickle.load(file)
    else:
        raise FileNotFoundError(f"{ptspath} is missing")

    for i, dp in pklpts.items():

        c4d.cprint(str(i), "c", end=")  ")
        for p in dp:
            c4d.cprint(p.class_id, end="  ")
        c4d.cprint()
