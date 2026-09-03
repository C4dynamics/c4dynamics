# type: ignore

import os
import sys
import subprocess

sys.path.append(".")
import c4dynamics as c4d

# Open output file for writing
os.makedirs("tests/_out", exist_ok=True)
output_file = open(r"tests\_out\doctests_output.txt", "w")

# Redirect stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = output_file
sys.stderr = output_file

print(sys.executable)

packagefol = "c4dynamics"


skip_datasets = False  # True # (default = false)
skip_videos = False  # True   # (default = false)

skipto = None  # 'rotmat' #

for dirpath, _, filenames in os.walk(packagefol):
    if "__pycache__" in dirpath:
        continue
    c4d.cprint(f"dir: {dirpath}", "c")

    for file_name in filenames:
        if skipto is not None and file_name != skipto + ".py":
            continue
        if file_name == "registery.py":
            continue
        if file_name == "kalman_v0.py":
            continue
        if file_name == "kalman_v1.py":
            continue
        if file_name == "yolo3_tf.py":
            continue
        if file_name == "a.py":
            continue
        if file_name == "luenberger.py":
            continue
        if file_name == "lineofsight.py":
            continue
        if file_name == "vidgen.py":
            continue
        if file_name == "video_gen.py":
            continue
        if file_name == "_struct.py":
            continue
        if file_name == "images_loader.py":
            continue
        if file_name == "slides_gen.py":
            continue
        if file_name == "plottracks.py":
            continue
        if file_name == "printpts.py":
            continue
        if dirpath.replace("\\", "/").endswith("utils/use_cases"):
            # supporting modules for the documentation use-case examples --
            # exercised by their notebooks, not by doctests.
            continue
        if not file_name.endswith(".py"):
            continue
        if skip_datasets and file_name == "_manager.py":
            continue
        if skip_videos and file_name == "yolo3_opencv.py":
            continue
        if file_name == "__init__.py" and dirpath == "c4dynamics":
            continue

        # if file_name == 'yolo3_opencv.py' or file_name == '_manager.py' or file_name == 'animate.py':
        #   c4d.cprint('warning: yolo3, datasets, animate, are skipped!', 'r')
        #   continue

        # testfile = os.path.join(dirpath, file_name)

        # if dirpath == 'c4dynamics' and file_name == '__init__.py':
        #   subprocess.run([sys.executable, '-m', 'c4dynamics'])
        # else:
        #   subprocess.run([sys.executable, testfile])

        c4d.cprint(f"file: {file_name}", "c")
        result = subprocess.run(
            [sys.executable, os.path.join(dirpath, file_name)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output_file.write(result.stdout)


# Restore stdout and stderr
sys.stdout = original_stdout
sys.stderr = original_stderr
output_file.close()

print("All output saved to 'doctests_output.txt'")
