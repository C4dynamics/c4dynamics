import sys

# sys.path.append(".")

from c4dynamics.envs.mountain_car import mountain_car as mountain_car

if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
