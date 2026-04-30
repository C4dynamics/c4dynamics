<div align="center">
  <img src="https://github.com/C4dynamics/C4dynamics/blob/main/docs/source/_icon/c4dlogotext.svg">
</div>





<div align="center">
  <strong> Published in the Journal of Open Source Software (JOSS)</strong><br>
  <a href="https://doi.org/10.5281/zenodo.17931207">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17931207.svg">
  </a>
  <br>
  <a href="https://doi.org/10.21105/joss.08776">
    <img src="https://joss.theoj.org/papers/10.21105/joss.08776/status.svg">
  </a>
</div>

<br>





# Tsipor Dynamics

## Algorithms Engineering and Development



Tsipor (bird) Dynamics (c4dynamics) is the Python framework for state-space modeling and algorithm development.





![Static Badge](https://img.shields.io/badge/python-%20?style=for-the-badge&logo=python&color=white)
![PyPI - Version](https://img.shields.io/pypi/v/c4dynamics?style=for-the-badge&color=orange&link=https%3A%2F%2Fpypi.org%2Fproject%2Fc4dynamics%2F)
![GitHub deployments](https://img.shields.io/github/deployments/C4dynamics/C4dynamics/github-pages%20?style=for-the-badge&label=docs)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/c4dynamics/c4dynamics/run-tests.yml?style=for-the-badge&label=tests&link=https%3A%2F%2Fgithub.com%2FC4dynamics%2FC4dynamics%2Fblob%2Fmain%2F.github%2Fworkflows%2Frun-tests.yml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/C4dynamics/C4dynamics/paper.yml?style=for-the-badge&label=Paper)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/c4dynamics?style=for-the-badge&color=blue%20&link=https%3A%2F%2Fpepy.tech%2Fprojects%2Fc4dynamics%3FtimeRange%3DthreeMonths%26category%3Dversion%26includeCIDownloads%3Dtrue%26granularity%3Ddaily%26viewType%3Dline%26versions%3D2.0.3%252C2.0.1%252C2.0.0)



### **Stop starting from scratch every time you change systems**

**Same workflow. Different systems.**

> Most engineers rebuild everything.
> **c4dynamics keeps the structure fixed.**

---

## What is c4dynamics?

**c4dynamics** is a Python framework for building, simulating, estimating, and controlling physical systems
— without resetting your workflow every time the system changes.

It gives you one consistent way to:

* define a system
* simulate its evolution
* estimate its state
* design control

Across:

> robotics · aerospace · autonomous systems · navigation

---

## 🧪 Examples

**Real implementations of modeling, estimation, and control**

> These are not isolated demos.
> They all follow the same structure.



<div align="center">

<table>
<tr>

<td width="300">
<a href="https://c4dynamics.github.io/c4dynamics/programs/dof6sim.html">
    <div style="padding:16px;border-radius:12px;background:#f3f4f6;border:1px solid #d0d7de;">
    <img src="docs/source/_static/missdistance.png" width="100"><br><br>
    <b>6-DOF Simulation</b><br>
    <span style="color:#555;">Proportional navigation guidance</span>
    </div>
</a>
</td>

<td width="300">
<a href="https://c4dynamics.github.io/c4dynamics/programs/ballistic_coefficient.html">
    <div style="padding:16px;border-radius:12px;background:#f3f4f6;border:1px solid #d0d7de;">
    <img src="docs/source/_static/ballistic_trajectory.png" width="100"><br><br>
    <b>Extended Kalman Filter</b><br>
    <span style="color:#555;">Ballistic coefficient estimation</span>
    </div>
</a>
</td>

<td width="300">
<a href="https://c4dynamics.github.io/c4dynamics/programs/car_tracker.html">
    <div style="padding:16px;border-radius:12px;background:#f3f4f6;border:1px solid #d0d7de;">
    <img src="docs/source/_static/drifting_car_snapshot.png" width="100"><br><br>
    <b>Detection + Kalman Filter</b><br>
    <span style="color:#555;">Vehicle tracking</span>
    </div>
</a>
</td>

<td width="300">
<a href="https://c4dynamics.github.io/c4dynamics/programs/mpc_steering.html">
    <div style="padding:16px;border-radius:12px;background:#f3f4f6;border:1px solid #d0d7de;">
    <img src="docs/source/_static/mpc_diagram.png" width="100"><br><br>
    <b>Model Predictive Control</b><br>
    <span style="color:#555;">Vehicle steering</span>
    </div>
</a>
</td>

</tr>
</table>
</div>


---

## The switching problem

Switching systems shouldn’t feel like starting over.

But it does:

* new models
* new simulation structure
* new estimation logic
* new control pipeline

You don’t just learn new physics.

> **You rebuild everything.**

---

## The workflow

> **Keep the workflow. Change the physics.**

c4dynamics enforces a consistent structure:

```
define → simulate → estimate → control
```

So when the system changes:

> your thinking doesn’t.

---

## Core principle

> **Physics first. Programming second.**

* Code implements
* Models define reality
* Algorithms follow structure

---

## What you get

* state-based modeling primitives
* simulation infrastructure
* Kalman / Extended Kalman filters
* sensor and detection modules
* reinforcement learning environments
* OpenCV / Open3D integration
* Monte Carlo simulation support

---


## Who this is for

* control engineers
* robotics engineers
* aerospace engineers
* autonomy developers

Especially if you’ve felt:

> “I know this stuff… but I don’t use it.”

---

## Quickstart

```python
>>> import c4dynamics as c4d
```

# define system
```python
s = c4d.state(y=1, vy=0.5)
```

# simulate
```python
F = [[1, 1],
     [0, 1]]

s.X += F @ s.X
s.store(t=1)
```



## Requirements
- 3.8 <= Python < 3.13
- Required packages are listed in [requirements.txt](requirements.txt)


---

## Installation

For detailed instructions on installing c4dynamics, including setup for virtual environments, Python version requirements, and troubleshooting, refer to the [c4dynamics setup guide](c4dynamics_setup.ipynb).

* [PIP](https://pypi.org/project/c4dynamics/)

```
>>> pip install c4dynamics
```


* [GitHub](https://github.com/c4dynamics/c4dynamics)

To run the latest GitHub version, download the repo and install required packages:

```
>>> pip install -r requirements.txt
```


---

## Documentation

📘 https://c4dynamics.github.io/c4dynamics/

* concepts
* API
* examples
* tutorials

---

## Contributing

This is not just a library.

It’s a shared way of building systems.

* build examples
* improve structure
* explore new systems

---

## Support
If you encounter problems, have questions, or would like to suggest improvements,
please open an Issue in this repository.


---

> **New system. Same workflow.**

---
