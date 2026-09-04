"""

`c4dynamics` provides an API to third party object detection models.


.. list-table::
  :header-rows: 0

  * - :class:`YOLOv3 <c4dynamics.detectors.yolo3_opencv.yolov3>`
    - Realtime object detection model based on YOLO (You Only Look Once)
      approach with 80 pre-trained COCO classes


"""

import sys

# sys.path.append(".")

from c4dynamics.detectors.yolo3_opencv import yolov3 as yolov3

if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
