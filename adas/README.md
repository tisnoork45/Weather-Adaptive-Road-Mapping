# ADAS Module

## Author
Darshpreet Singh

## Purpose
This module performs Advanced Driver Assistance System (ADAS) functions for the Weather Adaptive Road Mapping project.

## Components

### tracker.py
Tracks objects across frames and stores depth history.

### relative_velocity.py
Estimates relative velocity of detected objects using depth changes between frames.

### ttc_calculator.py
Calculates Time To Collision (TTC).

Formula:

TTC = Distance / Relative Velocity

### risk_engine.py
Generates safety decisions based on:

- Distance
- Relative Velocity
- TTC

Outputs:

- SAFE
- CAUTION
- STOP
- EMERGENCY

## Workflow

Object Detection
→ Tracking
→ Relative Velocity
→ TTC Calculation
→ Risk Assessment
→ Final Driving Decision