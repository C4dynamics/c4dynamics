# type: ignore

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

try:
    import cv2

    has_cv2 = True
except ImportError:
    has_cv2 = False


import os

import sys

sys.path.append(".")
from c4dynamics.detectors import yolov3
from c4dynamics import pixelpoint

MODEL_SIZE = (416, 416, 3)


@unittest.skipIf(not has_cv2, "opencv-python not installed")
class TestYoloV3(unittest.TestCase):

    @patch("c4dynamics.datasets.nn_model")
    @patch("os.path.exists")
    @patch("cv2.dnn.readNet")
    def setUp(self, mock_readNet, mock_exists, mock_nn_model):
        # Mock the path to weights for initialization
        home_folder = os.path.expanduser("~")
        mock_nn_model.return_value = os.path.join(
            home_folder, "AppData\\Local\\c4data\\yolov3.weights"
        )
        mock_exists.return_value = True

        # Create a MagicMock instance for net
        mock_net = MagicMock()
        mock_net.getLayerNames.return_value = ["layer1", "layer2", "layer3"]  # Example layer names
        mock_net.getUnconnectedOutLayers.return_value = [
            1,
            2,
            3,
        ]  # Example indices for unconnected layers
        mock_readNet.return_value = mock_net  # Return the mocked net

        self.yolo = yolov3()
        self.sample_frame = np.zeros((MODEL_SIZE[0], MODEL_SIZE[1], 3), dtype=np.uint8)

    @patch("c4dynamics.datasets.nn_model")
    @patch("os.path.exists")
    @patch("cv2.dnn.readNet")
    def test_initialization_default_weights(self, mock_readNet, mock_exists, mock_nn_model):
        home_folder = os.path.expanduser("~")
        mock_nn_model.return_value = os.path.join(
            home_folder, "AppData\\Local\\c4data\\yolov3.weights"
        )
        mock_exists.return_value = True
        mock_net = MagicMock()
        mock_net.getLayerNames.return_value = ["layer1", "layer2", "layer3"]
        mock_net.getUnconnectedOutLayers.return_value = [1, 2, 3]
        mock_readNet.return_value = mock_net
        yolo = yolov3()

        self.assertGreater(len(yolo.ln), 0)
        self.assertIsInstance(yolo.net, MagicMock)
        self.assertTrue(hasattr(yolo, "ln"))
        self.assertGreater(len(yolo.ln), 0)

    @patch("c4dynamics.datasets.nn_model")
    @patch("os.path.exists")
    @patch("cv2.dnn.readNet")
    def test_initialization_invalid_weights_path(self, mock_readNet, mock_exists, mock_nn_model):
        # Mock os.path.exists to return False for the invalid path
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            yolov3(weights_path="invalid/path/to/weights")

    @patch("c4dynamics.datasets.nn_model")
    @patch("os.path.exists")
    @patch("cv2.dnn.readNet")
    def test_threshold_getters_setters(self, mock_readNet, mock_exists, mock_nn_model):
        home_folder = os.path.expanduser("~")
        mock_nn_model.return_value = os.path.join(
            home_folder, "AppData\\Local\\c4data\\yolov3.weights"
        )
        mock_exists.return_value = True
        mock_net = MagicMock()
        mock_net.getLayerNames.return_value = ["layer1", "layer2", "layer3"]
        mock_net.getUnconnectedOutLayers.return_value = [1, 2, 3]
        mock_readNet.return_value = mock_net
        yolo = yolov3()

        yolo.nms_th = 0.6
        self.assertEqual(yolo.nms_th, 0.6)
        yolo.confidence_th = 0.7
        self.assertEqual(yolo.confidence_th, 0.7)

    @patch("cv2.dnn.blobFromImage")
    @patch("cv2.dnn.readNet")
    @patch("os.path.exists")
    @patch("c4dynamics.datasets.nn_model")
    def test_detect(self, mock_nn_model, mock_exists, mock_readNet, mock_blobFromImage):
        home_folder = os.path.expanduser("~")
        mock_nn_model.return_value = os.path.join(
            home_folder, "AppData\\Local\\c4data\\yolov3.weights"
        )
        mock_exists.return_value = True
        mock_net = MagicMock()
        mock_net.getLayerNames.return_value = ["layer1", "layer2", "layer3"]
        mock_net.getUnconnectedOutLayers.return_value = [1, 2, 3]
        mock_net.forward.return_value = [np.random.rand(1, 85)]
        mock_readNet.return_value = mock_net
        mock_blobFromImage.return_value = np.zeros((1, 416, 416, 3), dtype=np.float32)

        yolo = yolov3()
        sample_frame = np.zeros((MODEL_SIZE[0], MODEL_SIZE[1], 3), dtype=np.uint8)
        points = yolo.detect(sample_frame)

        self.assertIsInstance(points, list)
        for point in points:
            self.assertIsInstance(point, pixelpoint)
            self.assertTrue(hasattr(point, "class_id"))
            self.assertTrue(hasattr(point, "fsize"))

    @patch("cv2.dnn.readNet")
    @patch("os.path.exists")
    @patch("c4dynamics.datasets.nn_model")
    def test_detect_empty_frame(self, mock_nn_model, mock_exists, mock_readNet):
        home_folder = os.path.expanduser("~")
        mock_nn_model.return_value = os.path.join(
            home_folder, "AppData\\Local\\c4data\\yolov3.weights"
        )
        mock_exists.return_value = True
        mock_net = MagicMock()
        mock_net.getLayerNames.return_value = ["layer1", "layer2", "layer3"]
        mock_net.getUnconnectedOutLayers.return_value = [1, 2, 3]
        mock_readNet.return_value = mock_net

        yolo = yolov3()
        sample_frame = np.zeros((MODEL_SIZE[0], MODEL_SIZE[1], 3), dtype=np.uint8)
        points = yolo.detect(sample_frame)
        self.assertEqual(points, [])


if __name__ == "__main__":
    unittest.main(failfast=True)
