import os
import sys

# optional vision modules
try:
    import cv2
except ImportError:
    # vision modules are unavailable
    pass

from importlib.resources import files
from typing import Optional

import numpy as np

# sys.path.append(".")
from c4dynamics import c4d, pixelpoint

MODEL_SIZE = (416, 416, 3)


class yolov3:
    """
    YOLO: Real-Time Object Detection


    :class:`yolov3` is a YOLOv3 (You Only Look Once) object detection model.
    Though it is no longer the most accurate object detection algorithm,
    YOLOv3 is still a very good choice when you need real-time detection
    while maintaining excellent accuracy.



    YOLOv3 processes an entire image in a single forward pass,
    making it efficient for dynamic scenes.
    Its key strength lies the ability to simultaneously
    predict bounding box coordinates and class probabilities
    for multiple objects within an image.


    Parameters
    ==========
    weights_path : str, optional
        Path to the YOLOv3 weights file. Defaults None.


    See Also
    ========
    .filters
    .pixelpoint



    **Classes**


    Using YOLOv3 means
    object detection capability with the 80 pre-trained
    classes that come with the COCO dataset.


    The following 80 classes are available using COCO's pre-trained weights:

    .. admonition:: COCO dataset

        person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
        traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
        dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
        umbrella, handbag, tie, suitcase, frisbee, skis,snowboard, sports ball,
        kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
        bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
        sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
        couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote,
        keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book,
        clock, vase, scissors, teddy bear, hair drier, toothbrush



    .. figure:: /_architecture/yolo-object-detection.jpg

    *Figure 1*:
    Object Detection with YOLO using COCO pre-trained classes 'dog', 'bicycle', 'truck'.
    Read more at: `darknet-yolo <https://pjreddie.com/darknet/yolo/>`_.


    **Implementation (c4dynamics)**

    The :class:`yolov3` class abstracts the complexities of model initialization,
    input preprocessing, and output parsing.
    The :meth:`detect` method returns a
    :class:`pixelpoint <c4dynamics.states.lib.pixelpoint.pixelpoint>`
    for each detected object.
    The `pixelpoint` is a :mod:`predefined state class <c4dynamics.states.lib>`
    representing a data point in a video frame with an associated bounding box.
    Its methods and properties enhance the YOLOv3 output structure,
    providing a convenient data structure for handling tracking missions.




    **Installation**

    C4dynamics downloads
    the YOLOv3' weights file
    once at first call to :class:`yolov3` and saves it to the cache.
    For further details see :mod:`datasets <c4dynamics.datasets>`.
    Alternatively, the user can provide a path to his
    own weights file using the parameter `weights_path`.


    **Construction**

    A YOLOv3 detector instance is created by making a direct call
    to the `yolov3` constructor:

    .. code::

        >>> from c4dynamics.detectors import yolov3
        >>> yolo3 = yolov3()
        Fetched successfully


    Initialization of the instance does not require any mandatory parameters.



    Example
    =======

    The following snippet initializes the YOLOv3 model and
    runs the `detect()` method on an image containing four airplanes.
    The example uses the `datasets` module from `c4dynamics` to fetch an image.
    For further details, see :mod:`c4dynamics.datasets`.


    Import required packages:

    .. code::

        >>> import cv2
        >>> import c4dynamics as c4d
        >>> from matplotlib import pyplot as plt

    Load YOLOv3 detector:

    .. code::

        >>> yolo3 = c4d.detectors.yolov3()
        Fetched successfully

    Fetch and read the image:

    .. code::

        >>> imagepath = c4d.datasets.image('planes')
        Fetched successfully
        >>> img = cv2.imread(imagepath)


    Run YOLOv3 detector on an image:

    .. code::

        >>> pts = yolo3.detect(img)


    Now `pts` consists of
    :class:`pixelpoint <c4dynamics.states.lib.pixelpoint.pixelpoint>`
    instances for each object detected in the frame.
    Let's use the properties and methods of the `pixelpoint` class to
    view the attributes of the detected objects:


    .. code::

        >>> def ptup(n): return '(' + str(n[0]) + ', ' + str(n[1]) + ')'
        >>> print('{:^10} | {:^10} | {:^16} | {:^16} | {:^10} | {:^14}'.format('center x', 'center y', 'box top-left', 'box bottom-right', 'class', 'frame size')) # doctest: +IGNORE_OUTPUT
        >>> for p in pts:
        ...   print('{:^10} | {:^10} | {:^16} | {:^16} | {:^10} | {:^14}'.format(p.x, p.y, ptup(p.box[0]), ptup(p.box[1]), p.class_id, ptup(p.fsize)))     # doctest: +IGNORE_OUTPUT
        ...   cv2.rectangle(img, p.box[0], p.box[1], [0, 0, 0], 2)      # +IGNORE_OUTPUT
        ...   point = (int((p.box[0][0] + p.box[1][0]) / 2 - 75), p.box[1][1] + 22)     # doctest: +IGNORE_OUTPUT
        ...   cv2.putText(img, p.class_id, point, cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)     # doctest: +IGNORE_OUTPUT
        center x  |  center y  |   box top-left   | box bottom-right |   class    |   frame size
          615     |    295     |    (562, 259)    |    (668, 331)    | aeroplane  |  (1280, 720)
          779     |    233     |    (720, 199)    |    (838, 267)    | aeroplane  |  (1280, 720)
          635     |    189     |    (578, 153)    |    (692, 225)    | aeroplane  |  (1280, 720)
          793     |    575     |    (742, 540)    |    (844, 610)    | aeroplane  |  (1280, 720)


    .. code::

        >>> plt.figure() # doctest: +IGNORE_OUTPUT
        >>> plt.axis(False) # doctest: +IGNORE_OUTPUT
        >>> plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/yolov3/intro.png


    """

    class_names = (
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic",
        "light",
        "fire",
        "hydrant",
        "stop",
        "sign",
        "parking",
        "meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports",
        "ball",
        "kite",
        "baseball",
        "bat",
        "baseball",
        "glove",
        "skateboard",
        "surfboard",
        "tennis",
        "racket",
        "bottle",
        "wine",
        "glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot",
        "dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell",
        "phone",
        "microwave",
    )

    _nms_th = 0.5
    _confidence_th = 0.5

    def __init__(self, weights_path: Optional[str] = None) -> None:

        errormsg = ""
        if weights_path is None:
            weights_path = c4d.datasets.nn_model("YOLOv3")
            errormsg = "Try to clear the cache by 'c4dynamics.datasets.clear_cache()'"

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"The file 'yolov3.weights' does not " f"exist in: '{weights_path}'. {errormsg}"
            )

        # with resources.path("c4dynamics.detectors", "yolov3.cfg") as cfgpath:
        cfgpath = files("c4dynamics.detectors") / "yolov3.cfg"
        self.net = cv2.dnn.readNet(str(cfgpath), weights_path)

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ln = self.net.getLayerNames()
        self.ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    #   with open(coconames, 'r') as f:
    #     self.class_names = f.read().strip().split('\n')

    # self.__dict__.update(kwargs)

    @property
    def nms_th(self) -> float:
        """
        Gets and sets the Non-Maximum Suppression (NMS) threshold.

        Objects with confidence scores below this threshold are suppressed.



        Parameters
        ----------
        nms_th : float
            The new threshold value for NMS during object detection.
            Defaults: `nms_th = 0.5`.

        Returns
        -------
        nms_th : float
            The threshold value used for NMS during object detection.
            Objects with confidence scores below this threshold are suppressed.



        Example
        -------

        Import required packages:

        .. code::

            >>> import c4dynamics as c4d
            >>> from matplotlib import pyplot as plt
            >>> import cv2


        Fetch 'planes.png' using the c4dynamics' datasets module (see :mod:`c4dynamics.datasets`):

        .. code::

            >>> impath = c4d.datasets.image('planes')
            Fetched successfully


        Load YOLOv3 detector and set 3 NMS threshold values to compare:

        .. code::

            >>> yolo3 = c4d.detectors.yolov3()
            Fetched successfully
            >>> nms_thresholds = [0.1, 0.5, 0.9]


        Run the detector on each threshold:

        .. code::

            >>> _, axs = plt.subplots(1, 3)
            >>> for i, nms_threshold in enumerate(nms_thresholds):
            ...   yolo3.nms_th = nms_threshold
            ...   img = cv2.imread(impath)
            ...   pts = yolo3.detect(img)
            ...   for p in pts:
            ...     cv2.rectangle(img, p.box[0], p.box[1], [0, 255, 0], 2) # doctest: +IGNORE_OUTPUT
            ...   axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # doctest: +IGNORE_OUTPUT
            ...   axs[i].set_title(f"NMS Threshold: {nms_threshold}", fontsize = 6)
            ...   axs[i].axis('off') # doctest: +IGNORE_OUTPUT


        .. figure:: /_examples/yolov3/nms_th.png


        A high value (0.9) for the Non-Maximum Suppression (NMS) threshold here
        leads to an increased number of bounding boxes around a single object.
        When the NMS threshold is high, it means that a significant overlap is
        required for two bounding boxes to be considered redundant,
        and one of them will be suppressed.
        To address this issue, it's essential to choose an appropriate
        NMS threshold based on the characteristics of your dataset and the
        level of overlap between objects.
        A lower NMS threshold (e.g., 0.4 or 0.5)
        is commonly used to suppress redundant boxes effectively
        while retaining accurate detections.
        Experimenting with different
        threshold values and observing their impact on the results is crucial
        for optimizing the performance of object detection models.


        """
        return self._nms_th

    @nms_th.setter
    def nms_th(self, val: float) -> None:
        self._nms_th = val

    @property
    def confidence_th(self) -> float:
        """
        Gets and sets the confidence threshold used in the object detection.

        Detected objects with confidence scores below this threshold are filtered out.



        Parameters
        ----------
        confidence_th : float
            The new confidence threshold for object detection.
            Defaults: `confidence_th = 0.5`.

        Returns
        -------
        confidence_th : float
            The confidence threshold for object detection.
            Detected objects with confidence scores below this threshold are filtered out.


        Example
        -------

        Import required packages:

        .. code::

            >>> import c4dynamics as c4d
            >>> from matplotlib import pyplot as plt
            >>> import cv2


        Fetch 'planes.png' using the c4dynamics' datasets module (see :mod:`c4dynamics.datasets`):

        .. code::

            >>> impath = c4d.datasets.image('planes')
            Fetched successfully


        Load YOLOv3 detector and set 3 confidence threshold values to compare:

        .. code::

            >>> yolo3 = c4d.detectors.yolov3()
            Fetched successfully
            >>> confidence_thresholds = [0.9, 0.95, 0.99]


        Run the detector on each threshold:

        .. code::

            >>> _, axs = plt.subplots(1, 3)
            >>> for i, confidence_threshold in enumerate(confidence_thresholds):
            ...   yolo3.confidence_th = confidence_threshold
            ...   img = cv2.imread(impath)
            ...   pts = yolo3.detect(img)
            ...   for p in pts:
            ...     cv2.rectangle(img, p.box[0], p.box[1], [0, 255, 0], 2) # doctest: +IGNORE_OUTPUT
            ...   axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # doctest: +IGNORE_OUTPUT
            ...   axs[i].set_title(f"Confidence Threshold: {confidence_threshold}", fontsize = 6)
            ...   axs[i].axis('off') # doctest: +IGNORE_OUTPUT


        .. figure:: /_examples/yolov3/confidence_th.png


        A single object being missed, particularly when setting the confidence threshold to 0.99,
        suggests that the model is highly confident in its predictions.
        This level of performance is typically achievable when the model
        has been trained on a diverse and representative dataset,
        encompassing a wide variety of object instances, backgrounds,
        and conditions.


        """

        return self._confidence_th

    @confidence_th.setter
    def confidence_th(self, val: float) -> None:
        self._confidence_th = val

    def detect(self, frame: np.ndarray) -> list[pixelpoint]:
        """
        Detects objects in a frame using the YOLOv3 model.

        At each call, the detector performs the following steps:

        1. Preprocesses the frame by creating a blob, normalizing pixel values, and swapping Red
           and Blue channels.
        2. Sets input to the YOLOv3 model and performs a forward pass to obtain detections.
        3. Extracts detected objects based on a confidence threshold, calculates bounding box
           coordinates, and filters results using Non-Maximum Suppression (NMS).


        Parameters
        ----------
        frame : numpy.array
            An input frame for object detection.

        Returns
        -------
        out : list[pixelpoint]
            A list of :class:`pixelpoint <c4dynamics.states.pixelpoint.pixelpoint>` objects
            representing detected objects,
            each containing bounding box coordinates and class label.


        Examples
        --------

        **Setup**

        Import required packages:

        .. code::

            >>> import cv2  # opencv-python
            >>> import c4dynamics as c4d
            >>> from matplotlib import pyplot as plt



        Fetch 'planes.png' and 'aerobatics.mp4' using the c4dynamics' datasets
        module (see :mod:`c4dynamics.datasets`):

        .. code::

            >>> impath = c4d.datasets.image('planes')
            Fetched successfully
            >>> vidpath = c4d.datasets.video('aerobatics')
            Fetched successfully



        Load YOLOv3 detector:

        .. code::

            >>> yolo3 = c4d.detectors.yolov3()
            Fetched successfully




        Let the auxiliary function:

        .. code::

            >>> def ptup(n): return '(' + str(n[0]) + ', ' + str(n[1]) + ')'



        **Object detection in a single frame**


        .. code::

            >>> img = cv2.imread(impath)
            >>> pts = yolo3.detect(img)
            >>> for p in pts:
            ...   cv2.rectangle(img, p.box[0], p.box[1], [0, 255, 0], 2) # doctest: +IGNORE_OUTPUT

        .. code::

            >>> plt.figure() # doctest: +IGNORE_OUTPUT
            >>> plt.axis(False) # doctest: +IGNORE_OUTPUT
            >>> plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # doctest: +IGNORE_OUTPUT


        .. figure:: /_examples/yolov3/single_image.png



        **Object detection in a video**

        .. code::

            >>> video_cap = cv2.VideoCapture(vidpath)
            >>> while video_cap.isOpened():
            ...   ret, frame = video_cap.read()
            ...   if not ret: break
            ...   pts = yolo3.detect(frame)
            ...   for p in pts:
            ...     cv2.rectangle(frame, p.box[0], p.box[1], [0, 255, 0], 2) # doctest: +IGNORE_OUTPUT
            ...     cv2.imshow('YOLOv3', frame)  # doctest: +IGNORE_OUTPUT
            ...   cv2.waitKey(10) # doctest: +IGNORE_OUTPUT

        .. figure:: /_examples/yolov3/aerobatics.gif




        **The output structure**


        The output of the detect() function is a list of :class:`pixelpoint <c4dynamics.states.pixelpoint.pixelpoint>` object.
        The :class:`pixelpoint <c4dynamics.states.pixelpoint.pixelpoint>` has unique attributes to manipulate the detected object class and
        bounding box.

        .. code::

            >>> print('{:^10} | {:^10} | {:^10} | {:^16} | {:^16} | {:^10} | {:^14}' # doctest: +IGNORE_OUTPUT
            ...             .format('# object', 'center x', 'center y', 'box top-left', 'box bottom-right', 'class', 'frame size'))
            >>> # main loop:
            >>> for i, p in enumerate(pts):
            ...   print('{:^10d} | {:^10.3f} | {:^10.3f} | {:^16} | {:^16} | {:^10} | {:^14}'
            ...         .format(i, p.x, p.y, ptup(p.box[0]), ptup(p.box[1]), p.class_id, ptup(p.fsize)))
            ...   cv2.rectangle(img, p.box[0], p.box[1], [0, 0, 0], 2)      # doctest: +IGNORE_OUTPUT
            ...   point = (int((p.box[0][0] + p.box[1][0]) / 2 - 75), p.box[1][1] + 22)
            ...   cv2.putText(img, p.class_id, point, cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)  # doctest: +IGNORE_OUTPUT
            # object  |  center x  |  center y  |   box top-left   | box bottom-right |   class    |  frame size
               0      |   0.584    |   0.376    |    (691, 234)    |    (802, 306)    | aeroplane  |  (1280, 720)
               1      |   0.457    |   0.473    |    (528, 305)    |    (642, 376)    | aeroplane  |  (1280, 720)
               2      |   0.471    |   0.322    |    (542, 196)    |    (661, 267)    | aeroplane  |  (1280, 720)
               3      |   0.546    |   0.873    |    (645, 588)    |    (752, 668)    | aeroplane  |  (1280, 720)

        .. code::

            >>> plt.figure()  # doctest: +IGNORE_OUTPUT
            >>> plt.axis(False) # doctest: +IGNORE_OUTPUT
            >>> plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # doctest: +IGNORE_OUTPUT

        .. figure:: /_examples/yolov3/outformat.png


        """
        #
        # Step 1: Preprocess the Frame
        #   - Create a blob (binary large object) from the input frame with the
        #       specified dimensions
        #   - Normalize pixel values to a range of 0 to 1
        #   - Specify the dimensions of the input layer of the YOLOv3 model
        #   - Swap Red and Blue channels (BGR to RGB)
        #   - Set crop to False to preserve the original aspect ratio
        ##
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (MODEL_SIZE[0], MODEL_SIZE[1]), swapRB=True, crop=False
        )

        #
        # Step 2: Set Input to the YOLOv3 Model and Perform Forward Pass
        #   - Set the blob as the input to the YOLOv3 model
        #   - Get the names of the output layers of the model
        #   - Perform a forward pass through the model to obtain detections
        #
        # The returning detection structure:
        #   1) x_center
        #   2) y_center
        #   3) width
        #   4) height
        #   5) confidence score
        #   6:end) probabilities for each class
        ##
        self.net.setInput(blob)
        detections = self.net.forward(self.ln)

        #
        # Step 3: Extract Detected Objects
        #   - Iterate through the detected objects in the forward pass results
        #   - Filter objects based on confidence threshold
        #   - Calculate bounding box coordinates and convert to integers
        #   - Append bounding box coordinates and class labels to respective lists
        ##
        raw = []  # xc, yc, w, h
        boxes = []  # top left x, top left y, width, height
        classIDs = []
        confidences = []
        fheight, fwidth = frame.shape[:2]

        for detection in detections:
            for d in detection:

                scores = d[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if (
                    scores[classID] > self._confidence_th
                ):  # Adjust the confidence threshold as needed

                    box = d[:4] * [
                        fwidth,
                        fheight,
                        fwidth,
                        fheight,
                    ]  # relative (xc, yc, w, h) to pixels
                    # (center_x, center_y, width, height) = box.astype('int')

                    x = box[0] - box[2] / 2  # top left x
                    y = box[1] - box[3] / 2  # top left y

                    boxes.append([x, y, box[2], box[3]])  # top left x, top left y, width, height
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    raw.append(d[:4])

        indices = np.array(cv2.dnn.NMSBoxes(boxes, confidences, self._confidence_th, self._nms_th))

        # box_out = []
        # class_out = []
        points_out = []

        if len(indices) > 0:
            # for i in indices.flatten():
            for i in indices.ravel():
                # (x, y) = (boxes[i][0], boxes[i][1])
                # (w, h) = (boxes[i][2], boxes[i][3])
                #               x top left, y top left,   x bottom right,           y bottom right
                # box_out.append([boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],
                # boxes[i][1] + boxes[i][3]])

                # points_out.append(pixelpoint(raw[i], self.class_names[classIDs[i]], (w, h)))
                pp = pixelpoint(
                    x=int(raw[i][0] * fwidth),
                    y=int(raw[i][1] * fheight),
                    w=int(raw[i][2] * fwidth),
                    h=int(raw[i][3] * fheight),
                )
                # pp.units = 'normalized'
                pp.fsize = (fwidth, fheight)
                pp.class_id = self.class_names[classIDs[i]]
                points_out.append(pp)

                # class_out.append(self.class_names[classIDs[i]])

        # box_out = np.array(box_out)

        return points_out  # box_out, class_out,


if __name__ == "__main__":

    from c4dynamics import rundoctests

    # current_module = sys.modules[__name__].yolov3
    current_module = sys.modules[__name__]

    try:
        import cv2
    except ImportError:
        # if hasattr(current_module, "yolov3"):
        current_module.yolov3.__doc__ = ""  # clears doctest examples
        current_module.yolov3.detect.__doc__ = ""  # clears doctest examples
        current_module.yolov3.confidence_th.__doc__ = ""  # clears doctest examples
        current_module.yolov3.nms_th.__doc__ = ""  # clears doctest examples

    rundoctests(sys.modules[__name__])
