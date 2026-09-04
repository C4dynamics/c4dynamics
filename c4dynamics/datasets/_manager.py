import hashlib
import os
import shutil
import sys
from typing import Optional

# sys.path.append(".")
from c4dynamics.datasets._registry import (
    CACHE_DIR,
    d3_f16_register,
    d3_register,
    image_register,
    nn_register,
    video_register,
)

imagesmap = {
    "planes": "planes.png",
    "planes.png": "planes.png",
    "triangle": "triangle.png",
    "triangle.png": "triangle.png",
}
videosmap = {
    "aerobatics": "aerobatics.mp4",
    "aerobatics.mp4": "aerobatics.mp4",
    "drifting_car.mp4": "drifting_car.mp4",
    "driftingcar": "drifting_car.mp4",
    "drifting_car": "drifting_car.mp4",
}
nnsmap = {
    "yolov3": "yolov3.weights",
    "yolo_v3": "yolov3.weights",
    "yolov3.weights": "yolov3.weights",
}
d3smap = {
    "f16": "f16",
    "f_16": "f16",
    "bunny": "bunny.pcd",
    "bunnymesh": "bunny_mesh.ply",
    "bunny_mesh": "bunny_mesh.ply",
    "bunny.pcd": "bunny.pcd",
    "bunnymesh.ply": "bunny_mesh.ply",
    "bunny_mesh.ply": "bunny_mesh.ply",
}


def image(image_name: str) -> str:
    """
    Fetches the path of an image from the local cache.

    `image` downloads and manages image files from `c4dynamics` datasets.


    .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - Image Name
        - Description

      * - planes
        - A 1280x720 snapshot of aerobatic airplanes (planes.png, 300KB)

      * - triangle
        - A 800x600 image of a triangle (triangle.png, 20KB)



    The images can be found at
    https://github.com/c4dynamics/c4dynamics/blob/main/datasets/images/


    Parameters
    ----------
    image_name : str
        The name of the image to download

    Returns
    -------
    out : str
        A path to the image file in the local cache


    Examples
    --------

    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> import matplotlib.image as mpimg
      >>> from matplotlib import pyplot as plt

    **planes**

    .. code::

      >>> impath = c4d.datasets.image('planes')
      Fetched successfully
      >>> img = mpimg.imread(impath) # read image
      >>> plt.imshow(img) # doctest: +IGNORE_OUTPUT
      >>> plt.axis('off') # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/datasets/planes.png


    **triangle**

    .. code::

      >>> impath = c4d.datasets.image('triangle')
      Fetched successfully
      >>> img = mpimg.imread(impath) # read image
      >>> plt.imshow(img)   # doctest: +IGNORE_OUTPUT
      >>> plt.axis('off')   # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/datasets/triangle.png

    """

    imagein = imagesmap.get(image_name.lower(), None)

    if imagein is None:
        raise KeyError(
            f"'{image_name}' is not an image in c4dynamics datasets."
            " Available images are: 'planes', 'triangle'."
        )

    filename = image_register.fetch(imagein)
    print("Fetched successfully")
    return filename


def video(video_name: str) -> str:
    """
    Fetches the path of a video from the local cache.

    `video` downloads and manages video files from `c4dynamics` datasets.


    .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - Video Name
        - Description

      * - planes
        - 10 seconds video file of aerobatic airplanes (aerobatics.mp4, 7MB)

      * - drifting_car
        - 9 seconds video file of a drifting car\\* (drifting_car.mp4, 10MB)


    The videos can be found at
    https://github.com/c4dynamics/c4dynamics/blob/main/datasets/videos/

    \\* Used by kind permission of `Abed Ismail <https://www.pexels.com/@abed-ismail>`_


    Parameters
    ----------
    video_name : str
        The name of the video to download

    Returns
    -------
    out : str
        A path to the video file in the local cache


    Examples
    --------

    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> import cv2


    Fetch and run:

    .. code::

      >>> vidpath = c4d.datasets.video('aerobatics')
      Fetched successfully
      >>> cap = cv2.VideoCapture(vidpath)
      >>> while cap.isOpened():
      ...   ret, frame = cap.read()
      ...   if not ret: break
      ...   cv2.imshow('aerobatics', frame)
      ...   cv2.waitKey(33) # doctest: +IGNORE_OUTPUT
      >>> cap.release()
      >>> cv2.destroyAllWindows()


    .. figure:: /_examples/datasets/aerobatics.gif



    """

    videoin = videosmap.get(video_name.lower(), None)

    if videoin is None:
        raise KeyError(
            f"'{video_name}' is not a video in c4dynamics datasets."
            " An available video is: 'aerobatics'."
        )

    filename = video_register.fetch(videoin)
    print("Fetched successfully")
    return filename


def nn_model(nn_name: str) -> str:
    """
    Fetches the path of a neural network model from the local cache.

    `nn_model` downloads and manages neural network files from `c4dynamics` datasets.


    .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - NN Name
        - Description

      * - YOLOv3
        - Pre-trained weights file (237 MB)



    YOLOv3 weights file can be found at
    https://pjreddie.com/media/files/yolov3.weights


    Parameters
    ----------
    nn_name : str
        The name of the neural network model to download

    Returns
    -------
    out : str
        A path to the neural network in the local cache


    Examples
    --------

    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> import cv2


    Print first 5 layers of YOLOv3:

    .. code::

      >>> impath = c4d.datasets.nn_model('yolov3')
      Fetched successfully
      >>> net = cv2.dnn.readNet(impath, 'c4dynamics/detectors/yolov3.cfg')
      >>> for i, layer in enumerate(net.getLayerNames()):
      ...   print(f"Layer {i}:\t{layer}")
      ...   if i > 4: break
      Layer 0:  conv_0
      Layer 1:  bn_0
      Layer 2:  leaky_1
      Layer 3:  conv_1
      Layer 4:  bn_1
      Layer 5:  leaky_2


    """

    nnin = nnsmap.get(nn_name.lower(), None)

    if nnin is None:
        raise KeyError(
            f"'{nn_name}' is not a neural network model in c4dynamics datasets."
            " An available model is: 'yolov3'."
        )

    filename = nn_register.fetch(nnin)
    print("Fetched successfully")
    return filename


def d3_model(d3_name: str) -> str:
    """
    Fetches the path of a 3D model from the local cache.


    `d3_model` downloads and manages 3D model files from `c4dynamics` datasets.


    .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - Model Name
        - Description

      * - bunny
        - Point cloud file of Stanford bunny (bunny.pcd, 0.4MB)

      * - bunny_mesh
        - Polygon file of Stanford bunny (bunny_mesh.ply, 3MB)

      * - F16
        - A folder of 10 stl files, representing the jet parts as fuselage, ailerons,
          cockpit, rudder and stabilators (10 files, total 3MB).


    The models can be found at
    https://github.com/c4dynamics/c4dynamics/blob/main/datasets/d3_models/



    Parameters
    ----------
    d3_name : str
        The name of the 3D model to download

    Returns
    -------
    out : str
        A path to the model in the local cache.
        For the `f16` model, the function returns a path
        to the folder including 10 files.


    Examples
    --------

    **Stanford bunny (point cloud)**

    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> import open3d as o3d
      >>> import os


    .. code::

      >>> bunnypath = c4d.datasets.d3_model('bunny')
      Fetched successfully
      >>> pcd = o3d.io.read_point_cloud(bunnypath)
      >>> print(pcd)
      PointCloud with 35947 points.
      >>> o3d.visualization.draw_geometries([pcd])

    .. figure:: /_examples/datasets/bunny.png


    **Stanford bunny (triangle mesh)**

    .. code::

      >>> mbunnypath = c4d.datasets.d3_model('bunny_mesh')
      Fetched successfully
      >>> ply = o3d.io.read_triangle_mesh(mbunnypath)
      >>> ply.compute_vertex_normals() # doctest: +IGNORE_OUTPUT
      >>> print(ply)
      TriangleMesh with 35947 points and 69451 triangles.
      >>> o3d.visualization.draw_geometries([ply])

    .. figure:: /_examples/datasets/bunny_mesh.png


    **F16 (10 stl files)**

    .. code::

      >>> f16path = c4d.datasets.d3_model('f16')
      Fetched successfully
      >>> model = []
      >>> for f in sorted(os.listdir(f16path)):
      ...   model.append(
      ...     o3d.io.read_triangle_mesh(
      ...       os.path.join(f16path, f)
      ...     ).compute_vertex_normals()
      ...   )
      >>> o3d.visualization.draw_geometries(model)


    .. figure:: /_examples/datasets/f16.png


    """

    d3in = d3smap.get(d3_name.lower(), None)

    if d3in is None:
        raise KeyError(
            f"'{d3_name}' is not a 3D model in c4dynamics datasets."
            " Available models are: 'bunny', 'bunny_mesh', 'f16'."
        )

    # returns path
    if d3in == "f16":
        d3_f16_register.fetch("Aileron_A_F16.stl")
        d3_f16_register.fetch("Aileron_B_F16.stl")
        d3_f16_register.fetch("Body_F16.stl")
        d3_f16_register.fetch("Cockpit_F16.stl")
        d3_f16_register.fetch("LE_Slat_A_F16.stl")
        d3_f16_register.fetch("LE_Slat_B_F16.stl")
        d3_f16_register.fetch("Rudder_F16.stl")
        d3_f16_register.fetch("Stabilator_A_F16.stl")
        filename = os.path.dirname(d3_f16_register.fetch("Stabilator_B_F16.stl"))

    else:
        filename = d3_register.fetch(d3in)

    print("Fetched successfully")
    return filename


def download_all() -> None:
    """
    Downloads all available datasets to the local cache.

    The `download_all` function fetches all predefined datasets,
    including images, videos, neural network models, and 3D models,
    and stores them in the local cache.

    This function is a convenient way to ensure that all necessary data files are locally
    available for use without having to download each individually.

    The datasets include:

    .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - Dataset Type
        - Included Files

      * - Images
        - 'planes.png', 'triangle.png'

      * - Videos
        - 'aerobatics.mp4', 'drifting_car.mp4'

      * - Neural Networks
        - 'yolov3.weights'

      * - 3D Models
        - 'bunny.pcd', 'bunny_mesh.ply', 'f16' (multiple STL files)

    This function ensures all resources are fetched into the cache directory,
    making them accessible for further processing.

    Examples
    --------

    **Download all datasets**

    .. code::

      >>> import c4dynamics as c4d

    .. code::

      >>> c4d.datasets.download_all()
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully

    After downloading, you can access each dataset using its
    corresponding function, such as `image`, `video`, `nn_model`, or `d3_model`.

    """
    image("planes")
    image("triangle")
    video("aerobatics")
    nn_model("yolov3")
    d3_model("bunny")
    d3_model("bunny_mesh")
    d3_model("f16")


def clear_cache(dataset: Optional[str] = None) -> None:
    """
    Deletes datasets from the local cache.

    If a dataset name is provided, the function deletes this alone.
    Otherwise, clears all the cache.


    Parameters
    ----------
    dataset : str, optional
        The name of the dataset to delete.



    Examples
    --------

    .. code::

      >>> import c4dynamics as c4d
      >>> from matplotlib import pyplot as plt
      >>> import matplotlib.image as mpimg
      >>> import os

    **Delete a dataset file**

    .. code::

      >>> # download and verify
      >>> impath = c4d.datasets.image('planes')
      Fetched successfully
      >>> print(os.path.exists(impath))
      True
      >>> # clear and verify
      >>> c4d.datasets.clear_cache('planes')
      >>> print(os.path.exists(impath))
      False

    **Clear all**

    .. code::

      >>> # download all
      >>> c4d.datasets.download_all()
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully
      Fetched successfully
      >>> # clear all and verify
      >>> c4d.datasets.clear_cache()
      >>> for root, dirs, files in os.walk(CACHE_DIR):
      ...   for file in files:
      ...     print(os.path.join(root, file))
      ...   for dir in dirs:
      ...     print(os.path.join(root, dir))

    """

    if dataset is None:
        for root, dirs, files in os.walk(CACHE_DIR):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
        return

    allmaps = imagesmap | videosmap | nnsmap | d3smap
    datain = allmaps.get(dataset.lower(), None)

    if datain is None:
        raise KeyError(f"'{dataset}' is not a dataset in c4dynamics.")

    datafile = os.path.join(CACHE_DIR, datain)
    if datain == "f16":
        shutil.rmtree(datafile)
    else:
        if os.path.exists(datafile):
            os.remove(datafile)


def sha256(filename: str) -> str:
    """
    Computes the SHA-256 hash of a file.

    Parameters
    ----------
    filename : str
        Path to the file for which the hash needs to be computed.

    Returns
    -------
    str
        The SHA-256 hash of the file.

    """

    # filehash = pooch.file_hash(r'C:\Users\odely\Downloads\yolov3 (1).weights')

    hash_sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_sha256.update(byte_block)
    return hash_sha256.hexdigest()


if __name__ == "__main__":

    current_module = sys.modules[__name__]

    try:
        import open3d as o3d  # OPEN3D AVAILABLE   # noqa: F401
    except ImportError:  # OPEN3D UNAVAILABLE
        # remove the doctests for d3_model if open3d is not available
        if hasattr(current_module, "d3_model"):
            current_module.d3_model.__doc__ = ""  # clears doctest examples

    try:
        import cv2  # noqa: F401
    except ImportError:
        if hasattr(current_module, "video"):
            current_module.video.__doc__ = ""  # clears doctest examples
        if hasattr(current_module, "nn_model"):
            current_module.nn_model.__doc__ = ""  # clears doctest examples

    # import doctest, contextlib
    # from c4dynamics import IgnoreOutputChecker, cprint

    # # Register the custom OutputChecker
    # doctest.OutputChecker = IgnoreOutputChecker

    # tofile = False
    # optionflags = doctest.FAIL_FAST

    # if tofile:
    #   with open(os.path.join('tests', '_out', 'output.txt'), 'w') as f:
    #     with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    #       result = doctest.testmod(optionflags = optionflags)
    # else:
    #   result = doctest.testmod(optionflags = optionflags)

    # if result.failed == 0:
    #   cprint(os.path.basename(__file__) + ": all tests passed!", 'g')
    # else:
    #   print(f"{result.failed}")

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
