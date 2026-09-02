Sensors
=======

The ``sensors`` module in c4dynamics provides
physical and vision-based sensor models for simulating real-world measurements.

It encompasses a variety of sensing modalities, from physical measurements to computer vision.
This gives simulated agents the “eyes and ears” they need to perceive and interpret the environment.


This module is designed to be flexible and extensible,
allowing you to integrate multiple sensor types into your
simulations while maintaining consistent interfaces for data acquisition and processing.


This section covers the following components:


Navigation Sensors
------------------
Generic navigation-sensor models —
:class:`gps <c4dynamics.sensors.navigation.gps>`,
:class:`imu <c4dynamics.sensors.navigation.imu>`, and
:class:`magnetometer <c4dynamics.sensors.navigation.magnetometer>` —
that map a *true* state vector to a noisy, biased *measurement*, following the
same pattern as the seeker and radar models above.

Unlike the seeker and radar (which operate on a :class:`rigidbody
<c4dynamics.states.lib.rigidbody.rigidbody>` origin and a target), the navigation
sensors are written against the plain 12-state vector

.. math::

  X = [x, y, z, v_x, v_y, v_z, \varphi, \theta, \psi, p, q, r]^T

(the same ordering used by `rigidbody` and by
:mod:`c4dynamics.controllers.quad_pid`), so they drop directly into an
EKF/UKF ``predict``/``update`` loop without an intermediate adapter. The
:class:`imu <c4dynamics.sensors.navigation.imu>` measures rates and inertial
acceleration — both derivatives of the state — so it takes the truth
`rigidbody` object itself rather than a bare vector, and keeps its own
previous-sample history internally between calls.

**Error model**

Every navigation sensor shares the same two-term error model:

- **Bias** — a constant offset, fixed at construction and unchanged between
  samples.
- **Noise** — a zero-mean Gaussian sample, redrawn independently at every call
  to ``measure``.

Passing ``isideal = True`` mutes both, so ``measure`` returns the noise-free,
bias-free truth.

.. list-table::
  :widths: 26 34 10 22
  :header-rows: 1

  * - Sensor
    - Measures
    - Dim
    - Default noise (:math:`1\sigma`)
  * - GPS
    - inertial position :math:`x, y, z`
    - 3
    - :math:`0.5\ m`
  * - IMU — gyroscope
    - body rates :math:`p, q, r`
    - 3
    - :math:`0.01\ rad/s`
  * - IMU — accelerometer
    - body-frame specific force :math:`a_x, a_y, a_z`
    - 3
    - :math:`0.05\ m/s^2`
  * - Magnetometer
    - heading (yaw) :math:`\psi`
    - 1
    - :math:`0.05\ rad`

**Measurement equations**

The GPS and gyroscope read states directly, and the magnetometer reads the yaw
state alone:

.. math::

  h_{gps}(X) = [x, y, z]^T

  h_{gyro}(X) = [p, q, r]^T

  h_{mag}(X) = \psi

The accelerometer is non-linear — it senses the gravity reaction plus the
vehicle's own coordinate acceleration, projected into the body frame:

.. math::

  h_{acc}(X) = [BI] \cdot \big(\dot{v} + [0,\ 0,\ g]^T\big)

where :math:`[BI]` is the body-from-inertial DCM formed by the Euler angles and
:math:`\dot{v}` is the inertial-velocity derivative (approximated by the
:class:`imu <c4dynamics.sensors.navigation.imu>` as a finite difference against
the previous sample).

The heading is returned without wrapping to :math:`[-\pi, \pi]`; a filter that
consumes it is responsible for wrapping its own innovation. For a worked
estimation-control pipeline built on all three models, see the
:doc:`Quadcopter EKF </programs/ekf_estimation/quad_ekf>` example.


Whether you are simulating an autonomous vehicle, a missile guidance loop, or a robotic system,
the Sensors module gives your models the “eyes and ears” they need to interact with the dynamic world.


YOLOv3 Class
------------
A real-time object detection interface based on the YOLOv3 architecture.
It provides bounding boxes, class predictions, and confidence scores,
enabling simulated agents to perceive and classify visual elements in their environment.

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




Seeker Class
------------
Models a generic seeker sensor used in guidance and tracking simulations.
It measures the azimuth and elevation angles through an error model,
simulating how onboard seekers detect and track targets.

**Functionality**

At each time step, the seeker returns measurements based on the true geometry relative to the target.

Let the relative coordinates in an arbitrary frame of reference:

.. math::

  dx = target.x - seeker.x

  dy = target.y - seeker.y

  dz = target.z - seeker.z


The relative coordinates in the seeker body frame are given by:

.. math::

  x_b = [BR] \cdot [dx, dy, dz]^T

where :math:`[BR]` is a
Body from Reference DCM (Direction Cosine Matrix)
formed by the seeker three Euler angles. See the `rigidbody` section below.

The azimuth and elevation measures are then the spatial angles:

.. math::

  az = tan^{-1}{x_b[1] \over x_b[0]}

  el = tan^{-1}{x_b[2] \over \sqrt{x_b[0]^2 + x_b[1]^2}}



Where:

- :math:`az` is the azimuth angle
- :math:`el` is the elevation angle
- :math:`x_b` is the target-radar position vector in radar body frame

.. figure:: /_architecture/skr_definitions.svg

  Fig-1: Azimuth and elevation angles definition




Radar Class
-----------
Simulates a configurable radar sensor, producing measurements such as range, azimuth, and elevation.
As a subclass of `Seeker`, radar measurements are passed through an error model to simulate real-world sensor imperfections.

**Radar vs Seeker**


The following table
lists the main differences between
:class:`seeker <c4dynamics.sensors.seeker.seeker>` and :class:`radar <c4dynamics.sensors.radar.radar>`
in terms of measurements and
default error parameters:



.. list-table::
  :widths: 22 13 13 13 13 13 13
  :header-rows: 1

  * -
    - Angles
    - Range
    - :math:`σ_{Bias}`
    - :math:`σ_{Scale Factor}`
    - :math:`σ_{Angular Noise}`
    - :math:`σ_{Range Noise}`

  * - Seeker
    - ✔️
    - ❌
    - :math:`0.1°`
    - :math:`5%`
    - :math:`0.4°`
    - :math:`--`

  * - Radar
    - ✔️
    - ✔️
    - :math:`0.3°`
    - :math:`7%`
    - :math:`0.8°`
    - :math:`1m`




See Also
--------

.. list-table::
  :header-rows: 0

  * - :class:`YOLOv3 <c4dynamics.detectors.yolo3_opencv.yolov3>`
    - Realtime object detection model based on YOLO (You Only Look Once) approach
      with 80 pre-trained COCO classes.
  * - :class:`seeker <c4dynamics.sensors.seeker.seeker>`
    - Direction detector.
  * - :class:`radar <c4dynamics.sensors.radar.radar>`
    - Range-direction detector.
  * - :class:`gps <c4dynamics.sensors.navigation.gps>`
    - Inertial position receiver.
  * - :class:`imu <c4dynamics.sensors.navigation.imu>`
    - Gyroscope + accelerometer.
  * - :class:`magnetometer <c4dynamics.sensors.navigation.magnetometer>`
    - Heading (yaw) sensor.



