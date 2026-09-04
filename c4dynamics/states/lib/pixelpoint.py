import sys

import numpy as np

# sys.path.append(".")
from c4dynamics.states.state import state

# from enum import Enum


class pixelpoint(state):
    """
    A data point in a video frame with a bounding box.

    :class:`pixelpoint` is a data structure for managing
    object detections in tracking missions.
    It provides properties and methods to
    conveniently interact with computer vision modules.

    The `pixelpoint` state variables are
    the object center and the bounding box:

    .. math::

      X = [x, y, w, h]^T

    - Center pixel, bounding box size.




    **Arguments**

    x : float or int
        The x-coordinate of the center of the object.

    y : float or int
        The y-coordinate of the center of the object.

    w : float or int
        The width of the bounding box.

    h : float or int
        The height of the bounding box.


    These variables use for streamlined operations through
    the state vector :attr:`X <c4dynamics.states.state.state.X>`.

    Parameters define the object and are not part of the state.
    For the `pixelpoint` class these are the frame size and the object classification.


    Parameters
    ==========

    fsize : (int, int)
        Frame size in pixels.

    class_id : str
        Class label associated with the object.


    **Construction**


    A `pixelpoint` instance is typically created
    when a new object is detected.

    For a given detection `d` with the following indices:

    - [0] : x-center, pixels
    - [1] : y-center, pixels
    - [2] : width, pixels
    - [3] : height, pixels
    - [4:end] : probabilities for each class of the list `class_names`

    and for frames with dimensions `(f_width, f_height)`, the following snippet
    constructs a `pixelpoint` and updates its properties.

    Import packages:

    .. code::

      >>> from c4dynamics import pixelpoint
      >>> import numpy as np


    Given a detection with arbitrary values:

    .. code::

      >>> d = [50, 50, 15, 25, 0.8, 0.1, 0.0, 0.05, 0.89]
      >>> f_width, f_height = 100, 100
      >>> class_names = ['dog', 'cat', 'horse', 'fox']

    Initialize a `pixelpoint` instance with the input detection:

    .. code::

      >>> pp = pixelpoint(x = d[0], y = d[1], w = d[2], h = d[3])
      >>> pp.fsize = (f_width, f_height)
      >>> pp.class_id = class_names[np.argmax(d[5:])]


    See Also
    ========
    .lib
    .state

    Examples
    ========

    **Setup and Preliminaries**


    Import required packages:

    .. code::

      >>> import cv2  # opencv-python
      >>> import numpy as np
      >>> import c4dynamics as c4d
      >>> from matplotlib import pyplot as plt



    Fetch 'planes.png' and 'triangle.png' using the c4dynamics'
    datasets module (see :mod:`c4dynamics.datasets`):

    .. code::

      >>> tripath = c4d.datasets.image('triangle')
      Fetched successfully
      >>> planspath = c4d.datasets.image('planes')
      Fetched successfully



    Define two auxiliary functions.
    The first, `tridetect()`, returns bounding boxes of the detected triangles:

    .. code::

      >>> def tridetect(img):
      ...   _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 255, 0)
      ...   contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      ...   bbox = []
      ...   for contour in contours:
      ...     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
      ...     if len(approx) == 3:
      ...       bbox.append(cv2.boundingRect(contour))
      ...   return bbox


    The second function, `ptup`, converts a tuple of two numbers into a formatted string:

    .. code::

      >>> def ptup(n):
      ...   return '(' + str(n[0]) + ', ' + str(n[1]) + ')'





    **Construction from image**

    .. code::

      >>> img = cv2.imread(tripath)
      >>> pp = c4d.pixelpoint(x = int(img.shape[1] / 2), y = int(img.shape[0] / 2), w = 100, h = 100)
      >>> pp.fsize = img.shape[:2]
      >>> pp.class_id = 'triangle'



    **Construction from detection**

    Given a frame with dimensions `(f_width, f_height)` and a detection `d` from an object detector:

    .. code::

      >>> pp = c4d.pixelpoint(x = d[0], y = d[1], w = d[2], h = d[3])
      >>> pp.fsize = (f_width, f_height)
      >>> pp.class_id = class_names[np.argmax(d[5:])]




    **Triangles detection**

    Run a triangles detector and create a `pixelpoint` object
    per each detected triangle.
    Use :attr:`box <c4dynamics.states.lib.pixelpoint.pixelpoint.box>`
    to draw bounding boxes:

    .. code::

      >>> img = cv2.imread(tripath)
      >>> triangles = tridetect(img)


    .. code::

      >>> print('{:^10} | {:^10} | {:^16} | {:^16} | {:^10} | {:^14}'.format('center x', 'center y', 'box top-left', 'box bottom-right', 'class', 'frame size')) # doctest: +IGNORE_OUTPUT
      >>> # iterate over the detected triangles:
      >>> for tri in triangles: # doctest: +IGNORE_OUTPUT
      ...   pp = c4d.pixelpoint(x = int(tri[0] + tri[2] / 2), y = int(tri[1] + tri[3] / 2), w = tri[2], h = tri[3])
      ...   pp.fsize = img.shape[:2]
      ...   pp.class_id = 'triangle'
      ...   print('{:^10.1f} | {:^10.1f} | {:^16} | {:^16} | {:^10} | {:^14}'.format(pp.x, pp.y, ptup(pp.box[0]), ptup(pp.box[1]), pp.class_id, ptup(pp.fsize)))
      ...   cv2.rectangle(img, pp.box[0], pp.box[1], [0, 255, 0], 2)
      center x  |  center y  |   box top-left   | box bottom-right |   class    |   frame size
       399.0    |   274.0    |    (184, 117)    |    (614, 431)    |  triangle  |   (600, 800)

    .. code::

      >>> plt.figure() # doctest: +IGNORE_OUTPUT
      >>> plt.axis(False) # doctest: +IGNORE_OUTPUT
      >>> plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # doctest: +IGNORE_OUTPUT


    .. figure:: /_examples/pixelpoint/triangle.png







    **C4dynamics' YOLOv3 detector**

    The method :meth:`detect <c4dynamics.detectors.yolo3_opencv.yolov3.detect>`
    of the class :class:`yolov3 <c4dynamics.detectors.yolo3_opencv.yolov3>`
    returns a list of `pixelpoint` for the detected objects in an image.
    Print the output per a detected object and view the final image:


    .. code::

      >>> img = cv2.imread(planspath)
      >>> yolo3 = c4d.detectors.yolov3()
      Fetched successfully
      >>> pts = yolo3.detect(img)


    .. code::

      >>> # prepare for printing properties:
      >>> print('{:^10} | {:^10} | {:^16} | {:^16} | {:^10} | {:^14}'.format('center x', 'center y', 'box top-left', 'box bottom-right', 'class', 'frame size'))  # doctest: +IGNORE_OUTPUT
      >>> for p in pts:   # doctest: +IGNORE_OUTPUT
      ...   print('{:^10.1f} | {:^10.1f} | {:^16} | {:^16} | {:^10} | {:^14}'.format(p.x, p.y, ptup(p.box[0]), ptup(p.box[1]), p.class_id, ptup(p.fsize)))
      ...   cv2.rectangle(img, p.box[0], p.box[1], [0, 255, 0], 2)
      ...   point = (int((p.box[0][0] + p.box[1][0]) / 2 - 75), p.box[1][1] + 22)
      ...   cv2.putText(img, p.class_id, point, cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
      center x  |  center y  |   box top-left   | box bottom-right |   class    |  frame size
       615.0    |   295.0    |    (562, 259)    |    (668, 331)    | aeroplane  |  (1280, 720)
       779.0    |   233.0    |    (720, 199)    |    (838, 267)    | aeroplane  |  (1280, 720)
       635.0    |   189.0    |    (578, 153)    |    (692, 225)    | aeroplane  |  (1280, 720)
       793.0    |   575.0    |    (742, 540)    |    (844, 610)    | aeroplane  |  (1280, 720)


    .. code::

      >>> plt.figure()  # doctest: +IGNORE_OUTPUT
      >>> plt.axis(False)  # doctest: +IGNORE_OUTPUT
      >>> plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # doctest: +IGNORE_OUTPUT


    .. figure:: /_examples/pixelpoint/yolov3.png

    """

    x: float
    y: float
    w: float
    h: float

    def __init__(self, x=0, y=0, w=0, h=0):

        ppargs = {}
        ppargs.setdefault("x", x)
        ppargs.setdefault("y", y)
        ppargs.setdefault("w", w)
        ppargs.setdefault("h", h)

        # super().__init__(x = bbox[0], y = bbox[1], w = bbox[2], h = bbox[3])
        super().__init__(**ppargs)
        # NOTE what's the ultimate state? with velocities or not?
        #       they can be added later.
        # but if in anyway they are added, why not to introduce them here?
        #

        # self._units = 'pixels'

        # NOTE still not sure whats the best idea to init of these properties:
        # on the one hand they are not part of he state and shoudlnt be provdied as variables
        # on the other they are necessary for methods like box.
        # also the filters inherit from the state class i.e. they are states
        # nontheless they are initialized not only with the state vairables but also
        # with parameters.
        self._frameheight = None
        self._framewidth = None
        self._class = None

        # - x (float): X-coordinate of the center of the bounding box in relative coordinates.
        # - y (float): Y-coordinate of the center of the bounding box in relative coordinates.

        # self._boxwidth = bbox[2]
        # ''' float; Width of the bounding box in a normalized format
        # (0 = left image edge, 1 = right image edge. '''

        # self._boxheight = bbox[3]
        # ''' float; Height of the bounding box in a normalized format
        # (0 = upper image edge, 1 = bottom image edge. '''

        # self._framewidth = framesize[0]
        # ''' int; Width of the frame in pixels. '''

        # self._frameheight = framesize[1]
        # ''' int; Height of the frame in pixels. '''

        # self.class_id = class_id
        # ''' string; Class label associated with the data point. '''

    # @property
    # def X(self):
    #   return super().X.astype(np.int64)

    # parameters

    @property
    def fsize(self):
        """

        Gets and sets the frame size.

        Parameters
        ----------
        fsize : tuple
            Size of the frame in pixels (width, height).
            - (width)  int : Frame width in pixels.
            - (height) int : Frame height in pixels.


        Returns
        -------
        out : tuple
            A tuple of the frame size in pixels (width, height).


        Raises
        ------
        ValueError
            If `fsize` doesn't have exactly two elements, a ValueError is raised.

        Examples
        --------
        For detailed usage,
        see the examples in the introduction to the :class:`pixelpoint` class.
        """

        # TODO any way add even line example to show how to use given the objects defined.
        # for all the other methods too.
        return (self._framewidth, self._frameheight)

    @fsize.setter
    def fsize(self, fsize):
        if len(fsize) != 2:
            raise ValueError("fsize must have exactly two elements.")

        self._framewidth = fsize[0]
        self._frameheight = fsize[1]

    @property
    def class_id(self):
        """
        Gets and sets the object classification.


        Parameters
        ----------
        class_id : str
            Class label associated with the data point.


        Returns
        -------
        out : str
            The current class label associated with the data point.


        Raises
        ------
        ValueError
            If `class_id` is not str, a ValueError is raised.


        Examples
        --------
        For detailed usage,
        see the examples in the introduction to the :class:`pixelpoint` class.

        """
        return self._class

    @class_id.setter
    def class_id(self, class_id):
        if not isinstance(class_id, str):
            raise TypeError("class_id must be an str object")

        self._class = class_id

    # properties

    @property
    def box(self):
        """
        Returns the box coordinates.

        The box coordinates are given by:
        `[(x top left, y top left), (x bottom right, y bottom right)]`

        Returns
        -------
        out : list[tuple]
            List containing two tuples representing
            top-left and bottom-right coordinates (in pixels).

        Examples
        --------
        For detailed usage,
        see the examples in the introduction to the :class:`pixelpoint` class.

        """
        # .. include:: ../../states.lib.pixelpoint.examples.rst

        # if self._units == 'normalized':
        #   if self._frameheight is None or self._framewidth is None:
        #     raise ValueError('When ''pixelpoint'' units are ''normalized'',
        # the property ''fsize'' '
        #                         'must be set first with the frame width and height.')

        #   x = int(self.x * self._framewidth)
        #   y = int(self.y * self._frameheight)

        #   w = self.w * self._framewidth
        #   h = self.h * self._frameheight

        # else: # units = pixels.

        x = self.x
        y = self.y

        w = self.w
        h = self.h

        # top left
        xtl = int(x - w / 2)
        ytl = int(y - h / 2)

        # bottom right
        xbr = int(x + w / 2)
        ybr = int(y + h / 2)

        return [(xtl, ytl), (xbr, ybr)]

    @property
    def Xpixels(self):
        """
        Returns the state vector in pixel coordinates.

        When pixelpoint.units are set to `normalized`, the method `Xpixels`
        is used to return the state vector in pixels.

        Returns
        -------
        out : numpy.int32
          A numpy array of the normalized coordinates :math:`[x, y, v_x, v_y]` transformed
          to pixel coordinates considering the specific dimensions of the image.


        Examples
        --------

        Import required packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> import cv2


        Settings and initialization:

        .. code::

          >>> imgpath = c4d.datasets.image('planes')
          Fetched successfully
          >>> img = cv2.imread(imgpath)
          >>> yolo3 = c4d.detectors.yolov3()
          Fetched successfully
          >>> pts = yolo3.detect(img)

        Main loop:

        .. code::

          >>> print('{:^10} | {:^12} | {:^12} | {:^12} | {:^12}'.format('# object', 'X normalized', 'Y normalized', 'X pixels', 'Y pixels'))  # doctest: +IGNORE_OUTPUT
          >>> for i, p in enumerate(pts):  # doctest: +IGNORE_OUTPUT
          ...     X = p.Xpixels
          ...     print('{:^10d} | {:^12.3f} | {:^12.3f} | {:^12d} | {:^12d}'.format(i, p.x, p.y, X[0], X[1]))
          # object | X normalized | Y normalized |   X pixels   |   Y pixels
            0      |    0.427     |    0.339     |     503      |     232
            1      |    0.411     |    0.491     |     484      |     336
            2      |    0.550     |    0.397     |     648      |     272
            3      |    0.507     |    0.916     |     598      |     627

        """
        # TODO complete with full state vector.

        # superx = super().X
        return np.array(
            [
                self.x * self._framewidth,  # x   # type: ignore
                self.y * self._frameheight,  # y   # type: ignore
                self.w * self._framewidth,  # w   # type: ignore
                self.h * self._frameheight,
            ],  # h  # type: ignore
            dtype=np.int32,
        )

    @staticmethod
    def boxcenter(box):
        # XXX seems like useless function and indeed is not in use anywhere.
        """

        Calculates the center coordinates of bounding boxes.

        Given a list of bounding boxes, this static method computes the center
        coordinates for each box.



        Parameters
        ----------
        out : list[box]
          List containing one pixelpoint.box or more. where
          every pixelpoint.box has two tuples
          representing top-left and bottom-right coordinates.

        Returns
        -------
        out : numpy.ndarray
            An array containing center coordinates for each bounding box in the
            format [[center_x1, center_y1], [center_x2, center_y2], ...].

        """

        return np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in box])

    @staticmethod
    def video_detections(vidpath, tf=None, storepath=False):
        import os
        import pickle
        import zlib

        import cv2

        import c4dynamics as c4d

        cap = cv2.VideoCapture(vidpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt = 1 / fps  # 1 / frame per second = the length of a single frame
        if tf is None:
            N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames count
            tf = N * dt
        f_frames = int(tf / dt)

        yolo3 = c4d.detectors.yolov3()
        yolo3.nms_th = 0.45

        detections = {}

        for _ in range(f_frames + 1):
            print(_)
            ret, frame = cap.read()
            if not ret:
                break
            crc32 = zlib.crc32(frame.tobytes())
            detections[crc32] = yolo3.detect(frame)

        if storepath:
            pklname = os.path.join(storepath,
                                   os.path.basename(vidpath)[:-4] + ".pkl")  # type: ignore
            with open(pklname, "wb") as file:
                pickle.dump(detections, file)
            print(f"detections stored at {pklname}")

        cap.release()

        return detections


if __name__ == "__main__":

    from c4dynamics import rundoctests

    # current_module = sys.modules[__name__].yolov3
    current_module = sys.modules[__name__]

    try:
        import cv2  # noqa: F401
    except ImportError:
        # if hasattr(current_module, "yolov3"):
        current_module.pixelpoint.__doc__ = ""  # clears doctest examples
        current_module.pixelpoint.Xpixels.__doc__ = ""  # clears doctest examples
        # current_module.yolov3.confidence_th.__doc__ = ""   # clears doctest examples
        # current_module.yolov3.nms_th.__doc__ = ""   # clears doctest examples

    rundoctests(sys.modules[__name__])
    # NOTE: message about skipped tests becuase it currently
    # says all passed actually no one was ran.
