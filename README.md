<p align="center">
  <img src="https://lab-engineering.actia.tn/wp-content/uploads/2021/02/ACTIA_3coul_RVB.png" alt="Actia Logo" width="200"/>
</p>

# ADAS with mmWave Radar Fusion

**Internship Project @ Actia Engineering Services**  

Advanced Driver Assistance System (ADAS) leveraging **mmWave radar fusion** for object detection, tracking, and scene understanding.

---

## Problem

Single radar-based systems in autonomous vehicles have a **narrow field of view**, limiting situational awareness.

---

## Solution

**Fusion of multiple radars** to enhance the field of view and provide a safer driving experience.

In this project, I work with **point cloud data from mmWave radars** to detect and track various objects on the road. By applying fusion techniques, I aim to improve environment understanding and significantly widen the radar coverage.

---

## Why mmWave Radars?

- Excellent at measuring **velocity** and **angle** of objects.
- Robust performance in adverse weather conditions such as **fog, rain, and dust**.
- Provide reliable data for **object detection and tracking**.

---

## Dataset

For this internship project, I chose to work with the **RadarScenes Dataset** because it contains **2D point clouds with velocity and RCS information** and is already labeled.

You can find and download the dataset here:  
[https://zenodo.org/records/4559821](https://zenodo.org/records/4559821)

---

## Proposed Workflow & Models

Below is the proposed workflow and the models I plan to explore, based on reviewing multiple research papers and comparing results:

![Workflow](./Img/Workflow.png)

---

## Current Status

Working on optimizing the Classification head.
