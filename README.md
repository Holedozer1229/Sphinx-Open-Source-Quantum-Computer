# Sphinx - Open Source Quantum Computer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

A quantum computing framework integrating scalar field dynamics and 4D spacetime geometry, inspired by the *Scalar Waze* theory.

## Features
- **4D Quantum Walks**: Simulate particle dynamics in curved spacetime.
- **M-Shift Operator**: Modify scalar field behavior in real-time.
- **Hardware Integration**: Control ion traps via serial communication.
- **Visualization**: 3D/4D plotting of quantum trajectories.

The Sphinx Quantum Computer is a sophisticated quantum computing system that utilizes OAM (Orbital Angular Momentum) modulated fiber optic arrays, trap ion gates, phosphorescent targets with quantum dots, and silicon drift detectors. This project simulates the quantum processes, including phase shifts, temporal displacement, and entanglement, using Qiskit and Python.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Components](#components)
5. [Simulation Details](#simulation-details)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

The Sphinx Quantum Computer project aims to model and simulate a quantum computing system that operates using advanced quantum hardware components. The simulation includes the following key features:

- Induction of phase shifts on virtual photons in a CTC (Closed Timelike Curve) fiber optic path.
- Utilization of trap ion gates for controlled interactions.
- Phosphorescent targets with quantum dots for state manipulation and measurement.
- Silicon drift detectors for entanglement and feedback.

## Installation

To run the Sphinx Quantum Computer simulation, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Holedozer1229/sphinx-quantum-computer.git
    cd sphinx-quantum-computer
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main simulation script is `sphinx_simulation.py`. To run the simulation, use the following command:

```bash
python sphinx_simulation.py
