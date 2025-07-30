#!/usr/bin/env python3
"""
Nairobi CBD Protest Navigation - Main Entry Point
Reinforcement Learning Project for Safe Protest Navigation

This file serves as the main entry point for running experiments,
demonstrations, and training different RL algorithms.
"""

import os
import sys
import argparse
import pygame
import numpy as np
from pathlib import Path
import time
import imageio
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from environment.custom_env import NairobiCBDProtestEnv
from environment.rendering import NairobiCBD3DRenderer, create_demo_visualization

def capture_opengl_frame(width, height):
    # Read pixels from OpenGL framebuffer
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    # Flip vertically (OpenGL's origin is bottom-left)
    image = np.flipud(image)
    # Convert to surface for imageio (drop alpha if needed)
    return image[:, :, :3]

def create_random_agent_demo(duration: int = 60, save_gif: bool = True, use_3d: bool = False):
    """
    Create a demonstration of the agent taking random actions in the environment.
    This satisfies the requirement for showing random actions without any training.
    
    Args:
        duration: Duration of demo in seconds
        save_gif: Whether to save frames as GIF
        use_3d: Whether to use 3D visualization
    """
    print("Creating Random Agent Demonstration...")
    print(f"Duration: {duration} seconds")
    print(f"3D Visualization: {'Enabled' if use_3d else 'Disabled'}")
    
    # Initialize Pygame
    pygame.init()
    
    # Initialize environment
    env = NairobiCBDProtestEnv(render_mode="human" if not use_3d else None)
    
    # Initialize renderer
    if use_3d:
        renderer = NairobiCBD3DRenderer()
        frames = []
    else:
        renderer = None
        if save_gif:
            frames = []
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    episode_count = 0
    total_reward = 0
    
    running = True
    clock = pygame.time.Clock()
    
    print("\nStarting random agent demo...")
    print("The agent will take random actions to demonstrate the environment")
    print("Press ESC or close window to stop early")
    
    try:
        while running and (time.time() - start_time) < duration:
            # Handle events
            if use_3d:
                running = renderer.handle_events()
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
            
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Render
            if use_3d:
                # Prepare state for 3D renderer
                env_state = {
                    'agent_pos': env.agent_pos,
                    'police_units': env.police_units,
                    'tear_gas_clouds': env.tear_gas_clouds,
                    'water_cannons': env.water_cannons,
                    'crowd_density_map': env.crowd_density_map,
                    'buildings': env.buildings,
                    'safe_zones': env.safe_zones,
                    'grid_width': env.grid_width,
                    'grid_height': env.grid_height
                }
                renderer.render_scene(env_state)
                
                # Capture frame for GIF
                if save_gif and step_count % 3 == 0:  # Every 3rd frame to reduce size
                    frame = capture_opengl_frame(renderer.width, renderer.height)
                    frames.append(frame)
            else:
                frame = env.render()
                if save_gif and step_count % 2 == 0:  # Every 2nd frame
                    frames.append(frame)
            
            # Reset if episode ends
            if done or truncated:
                episode_count += 1
                print(f"Episode {episode_count} completed. Steps: {step_count}, Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
            
            # Control frame rate
            clock.tick(30)
            
            # Print progress every 5 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and step_count % 150 == 0:
                print(f"Demo progress: {elapsed:.1f}/{duration}s - Steps: {step_count}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Save GIF if requested
        if save_gif and use_3d and frames:
            print("\nSaving 3D demonstration GIF...")
            gif_path = Path("media") / "random_agent_demo_3d.gif"
            gif_path.parent.mkdir(exist_ok=True)
            
            try:
                imageio.mimsave(
                    str(gif_path), 
                    frames, 
                    fps=15
                )
                print(f"3D GIF saved to: {gif_path}")
            except Exception as e:
                print(f"Failed to save 3D GIF: {e}")
        
        # Cleanup
        if use_3d and renderer:
            renderer.close()
        env.close()
        
        print(f"\nDemo completed!")
        print(f"Total episodes: {episode_count}")
        print(f"Total steps: {step_count}")
        print(f"Average steps per episode: {step_count / max(episode_count, 1):.1f}")


def run_environment_test():
    """Test the environment functionality"""
    print("Testing Nairobi CBD Protest Environment...")
    
    env = NairobiCBDProtestEnv()
    
    # Test environment creation
    print("✓ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset - Observation shape: {obs.shape}")
    print(f"  Initial observation: {obs}")
    
    # Test action space
    print(f"✓ Action space: {env.action_space} (6 discrete actions)")
    print("  Actions: 0=North, 1=South, 2=East, 3=West, 4=Stay, 5=Sprint")
    
    # Test observation space
    print(f"✓ Observation space: {env.observation_space}")
    print("  State components: [agent_x, agent_y, police_x, police_y, police_dist,")
    print("                     crowd_density, teargas, watercannon, safe_zone, strategy_timer]")
    
    # Test a few steps
    print("\nTesting environment dynamics...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {step+1}: Action={action}, Reward={reward:.2f}, Done={done}")
        
        if done:
            obs, info = env.reset()
            print("    Episode ended, environment reset")
    
    env.close()
    print("✓ Environment test completed successfully!")




def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Nairobi CBD Protest Navigation RL Project"
    )
    
    parser.add_argument(
        "--demo", action="store_true",
        help="Run random agent demonstration"
    )
    parser.add_argument(
        "--test", action="store_true", 
        help="Test environment functionality"
    )
    parser.add_argument(
        "--3d", dest="use_3d", action="store_true",
        help="Use 3D visualization (requires OpenGL)"
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Demo duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--no-gif", action="store_true",
        help="Don't save demo as GIF"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Create project directory structure"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("Nairobi CBD Protest Navigation - RL Project")
    print("=" * 60)
    
    try:
        # Test environment
        if args.test:
            run_environment_test()
            return
            
        # Run demonstration
        if args.demo:
            create_random_agent_demo(
                duration=args.duration,
                save_gif=not args.no_gif,
                use_3d=args.use_3d
            )
            return
        
        # Default: Show help and run basic demo
        if len(sys.argv) == 1:
            print("Welcome to the Nairobi CBD Protest Navigation RL Project!")
            print("\nAvailable commands:")
            print("  --demo          Run random agent demonstration")
            print("  --test          Test environment functionality") 
            print("  --3d            Use 3D visualization")
            print("  --setup         Create project structure")
            print("  --help          Show all options")
            print("\nRunning basic demo...")
            time.sleep(2)
            create_random_agent_demo(duration=30, save_gif=True, use_3d=False)
        else:
            parser.print_help()
            
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nMissing dependencies. Please install requirements:")
        print("pip install -r requirements.txt")
        print("\nOr create project structure first:")
        print("python main.py --setup")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Use --help for usage information")


if __name__ == "__main__":
    main()