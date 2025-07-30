# Nairobi CBD Protest Navigation - RL Project

## Overview
This project simulates a peaceful protester navigating Nairobi's CBD during a dynamic demonstration. The agent must avoid aggressive police units, tear-gas clouds, and water-cannon sprays while remaining within safe zones.

## Project Structure
```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   ├── rendering.py             # 3D visualization components
├── training/
│   ├── dqn_training.py          # DQN training script
│   ├── pg_training.py           # Policy gradient training
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models  
├── media/                       # Generated videos/GIFs
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Test environment: `python main.py --test`
3. Run random demo: `python main.py --demo --duration 30`
4. 3D visualization: `python main.py --demo --3d`

## Environment Details
- **Action Space**: 6 discrete actions (N/S/E/W/Stay/Sprint)
- **State Space**: 10-dimensional continuous observation
- **Rewards**: Survival (+1), avoiding police/hazards, reaching safe zones

## Training Algorithms
- **Value-Based**: DQN
- **Policy Gradient**: REINFORCE, PPO, Actor-Critic

## Visualization
- 2D top-down view with Pygame
- 3D immersive environment with PyOpenGL
- Real-time crowd density, tear gas, and water cannon effects