# Nairobi CBD Protest Navigation - Reinforcement Learning Project

## Overview

This project simulates a peaceful protester navigating through a highly dynamic and hazardous environment in **Nairobi's Central Business District (CBD)**. The agent must avoid aggressive police patrols, unpredictable tear gas dispersal, and water cannon blasts, while attempting to remain within safe protest zones and reach designated exits. Reinforcement Learning (RL) techniques are applied to train this agent under socially grounded, real-world inspired conditions.

>  Inspired by real protest dynamics, the project models both crowd behavior and hostile policing patterns, making it a unique testbed for risk-aware navigation under partial observability.

---

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   └── rendering.py             # 3D visualization using PyOpenGL
├── training/
│   ├── dqn_training.py          # DQN training manager
│   └── pg_training.py           # REINFORCE, PPO, A2C training manager
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models  
├── demos/                       # Generated videos and GIFs of agent behavior
├── main.py                      # Entry script for demos and environment tests
├── requirements.txt             # Project dependencies
└── README.md                    # You're here
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Environment (Random Agent)

```bash
python main.py --test
```

### 3. Run a Randomized Demo

```bash
python main.py --demo --duration 30
```

### 4. Launch 3D Visualization (Requires OpenGL)

```bash
python main.py --demo --3d
```

### Basic Usage

| Command                                        | Description           |
|------------------------------------------------|-----------------------|
| python main.py --train dqn                     | Train DQN agent       |
| python main.py --eval a2c --model best_a2c.zip | Evaluate saved model  |
| python main.py --demo --render 3d              | 3D visualization demo |
---

## Environment Details

* **Action Space**:
  5 discrete actions:

  * `0`: Move North
  * `1`: Move South
  * `2`: Move East
  * `3`: Move West
  * `4`: Sprint (Double movement in current direction)

* **Observation Space**:
  10-dimensional continuous vector including:

  * Agent position
  * Relative positions of nearest police
  * Crowd density
  * Hazard proximity (tear gas, water cannons)
  * Distance to exits and safe zones

* **Reward Structure Highlights**:

  * +100 for reaching exit
  * -100 for encountering hazards
  * Progressive survival reward (decays over time)
  * Penalty for inertia, wall collisions, or hazard zones
  * Bonus for safe exploration (coverage-based incentives)

---

## Trained Algorithms

* **Value-Based**:

  * Deep Q-Network (DQN)

* **Policy Gradient Methods**:

  * REINFORCE
  * Proximal Policy Optimization (PPO)
  * Advantage Actor-Critic (A2C)

Each method was trained on the same custom environment and evaluated under randomized unseen conditions.

---

## Visualizations

### Real-time Renderings:

* **Top-Down 2D View**: Basic grid layout using Pygame
* **3D View**: Interactive simulation using PyOpenGL

### Training Metrics:

* **Cumulative reward per episode**
* **Entropy loss and policy objective curves (PG methods)**
* **Success rates on unseen spawn points**

---

##  Evaluation Highlights

* **Robustness testing** on novel patrol routes, crowd configurations, and safe zone placements
* **Generalization metrics**: success rate, hazard avoidance, performance drop
* DQN demonstrated strongest generalization; A2C had highest overfitting despite best training scores.

---

## Future Work

* Implement **curriculum learning** to scale difficulty over time
* Add **intrinsic motivation** (e.g. curiosity-driven exploration)
* Explore **multi-agent training** to simulate protester group dynamics

---
