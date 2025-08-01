import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PoliceUnit:
    x: float
    y: float
    patrol_route: List[Tuple[float, float]]
    current_target: int
    speed: float
    strategy_timer: int
    
@dataclass
class TearGas:
    x: float
    y: float
    radius: float
    intensity: float
    duration: int
    
@dataclass
class WaterCannon:
    x: float
    y: float
    angle: float
    range: float
    width: float
    intensity: float
    duration: int

class NairobiCBDProtestEnv(gym.Env):
    """
    Custom Gymnasium environment simulating a peaceful protester navigating Nairobi's CBD
    during a dynamic demonstration while avoiding police units, tear gas, and water cannons.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: Tuple[int, int] = (100, 100)):
        super().__init__()
        
        # Environment parameters
        self.grid_width, self.grid_height = grid_size
        self.cell_size = 8  # pixels per grid cell for rendering
        self.step_size = 1.0
        self.sprint_multiplier = 2.0
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # State space: 10-dimensional continuous observation
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = self.grid_width * self.cell_size
        self.screen_height = self.grid_height * self.cell_size
        
        # Initialize environment components
        self._initialize_nairobi_cbd_layout()
        self._initialize_safe_zones()
        self._initialize_buildings()
        
        # Dynamic elements
        self.police_units: List[PoliceUnit] = []
        self.tear_gas_clouds: List[TearGas] = []
        self.water_cannons: List[WaterCannon] = []
        self.crowd_density_map = np.zeros((self.grid_width, self.grid_height))
        
        # Agent state
        self.agent_pos = np.array([0.0, 0.0])
        self.last_direction = np.array([1.0, 0.0])  # For sprint action
        self.steps_since_strategy_shift = 0
        
        # initialize visited map
        self.visited_map = np.zeros((self.grid_width, self.grid_height))
        
        self.start_pos = None # Initial agent position

        self.safe_zone_timer = 0
        self.last_safe_zone = None
        # Colors for rendering
        self.colors = {
            'background': (50, 50, 50),
            'road': (128, 128, 128),
            'building': (80, 80, 80),
            'safe_zone': (0, 255, 0, 100),
            'agent': (0, 100, 255),
            'police': (255, 0, 0),
            'tear_gas': (255, 165, 0, 128),
            'water_cannon': (0, 191, 255, 128),
            'crowd_low': (255, 255, 0, 50),
            'crowd_high': (255, 0, 0, 100)
        }
    
    def _initialize_nairobi_cbd_layout(self):
        """Initialize the basic layout of Nairobi CBD with roads and key landmarks"""
        self.roads = []
        self.landmarks = {}
        
        # Major roads in Nairobi CBD (simplified grid)
        # Horizontal roads
        for y in [20, 35, 50, 65, 80]:
            self.roads.extend([(x, y) for x in range(10, 90)])
        
        # Vertical roads  
        for x in [15, 30, 45, 60, 75]:
            self.roads.extend([(x, y) for y in range(10, 90)])
        
        # Key landmarks (simplified positions)
        self.landmarks = {
            'parliament': (25, 25),
            'city_hall': (45, 45),
            'uhuru_park': (65, 30),
            'central_park': (40, 65),
            'bus_station': (70, 70)
        }
    
    def _initialize_safe_zones(self):
        """Define safe zones where protesters can seek refuge"""
        self.safe_zones = [
            {'center': (20, 20), 'radius': 8},  # Near Parliament
            {'center': (65, 30), 'radius': 10}, # Uhuru Park area
            {'center': (75, 75), 'radius': 6},  # Shopping area
            {'center': (15, 80), 'radius': 5},  # Residential area
        ]
        
        # Exit points for successful episode completion
        self.exit_points = [
            (5, 50), (95, 50), (50, 5), (50, 95)  # Border exits
        ]
    
    def _initialize_buildings(self):
        """Define building locations that block movement - maze-like layout"""
        self.buildings = []
        
        # Create distinct building blocks with clear separation (maze-like pattern)
        building_areas = [
            # Government district (top-left)
            {'top_left': (12, 12), 'size': (8, 6)},   
            {'top_left': (25, 10), 'size': (6, 8)},   
            
            # Business district (center-left)
            {'top_left': (10, 30), 'size': (12, 8)},  
            {'top_left': (28, 32), 'size': (8, 10)},  
            
            # Shopping district (center)
            {'top_left': (42, 25), 'size': (10, 6)},  
            {'top_left': (45, 38), 'size': (6, 8)},   
            
            # Office towers (center-right)
            {'top_left': (60, 15), 'size': (8, 12)},  
            {'top_left': (72, 20), 'size': (6, 8)},   
            
            # Mixed development (bottom section)
            {'top_left': (15, 55), 'size': (10, 8)},  
            {'top_left': (35, 60), 'size': (8, 6)},   
            {'top_left': (50, 58), 'size': (6, 10)},  
            {'top_left': (65, 55), 'size': (12, 6)},  
            
            # Industrial area (bottom-right)
            {'top_left': (75, 70), 'size': (8, 10)},  
            
            # Residential blocks (edges)
            {'top_left': (10, 75), 'size': (6, 6)},   
            {'top_left': (85, 10), 'size': (6, 8)},   
        ]
        
        # Add buildings with 2-3 cell spacing for clear navigation paths
        for area in building_areas:
            top_left = area['top_left']
            size = area['size']
            for dx in range(size[0]):
                for dy in range(size[1]):
                    self.buildings.append((top_left[0] + dx, top_left[1] + dy))
    
    def _spawn_police_units(self, num_units: int = 5):
        """Spawn police units with patrol routes"""
        self.police_units.clear()
        
        patrol_routes = [
            [(25, 25), (35, 25), (35, 35), (25, 35)],  # Parliament area
            [(60, 40), (70, 40), (70, 60), (60, 60)],  # Central patrol
            [(15, 15), (85, 15), (85, 85), (15, 85)],  # Perimeter patrol
            [(40, 30), (50, 30), (50, 70), (40, 70)],  # Main street
            [(20, 50), (80, 50), (80, 80), (20, 80)]   # Southern patrol
        ]
        
        for i in range(min(num_units, len(patrol_routes))):
            route = patrol_routes[i]
            start_pos = route[0]
            self.police_units.append(PoliceUnit(
                x=start_pos[0],
                y=start_pos[1],
                patrol_route=route,
                current_target=1,
                speed=0.8,
                strategy_timer=0
            ))
    
    def _update_crowd_density(self):
        """Update crowd density map with concentrated dot patterns around landmarks"""
        self.crowd_density_map.fill(0.0)  # Clear previous crowd
        
        # Create concentrated crowd dots around landmarks
        crowd_spots = []
        
        # Generate crowd gathering points around landmarks
        for landmark, pos in self.landmarks.items():
            # Main gathering point
            crowd_spots.append({
                'center': pos, 
                'radius': 3, 
                'intensity': 0.9,
                'num_dots': 8
            })
            
            # Secondary gathering points around the main landmark
            for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
                offset_x = int(8 * math.cos(angle))
                offset_y = int(8 * math.sin(angle))
                secondary_pos = (pos[0] + offset_x, pos[1] + offset_y)
                
                # Ensure position is within bounds and not in buildings
                if (0 <= secondary_pos[0] < self.grid_width and 
                    0 <= secondary_pos[1] < self.grid_height and
                    not self._is_in_building(secondary_pos[0], secondary_pos[1])):
                    
                    crowd_spots.append({
                        'center': secondary_pos,
                        'radius': 2,
                        'intensity': 0.6,
                        'num_dots': 4
                    })
        
        # Dynamic crowd movement (protest march)
        time_factor = (self.current_step % 200) / 200.0
        march_center_x = 20 + int(40 * time_factor)  # Move across the city
        march_center_y = 45 + int(10 * math.sin(time_factor * 2 * math.pi))
        
        # Add marching crowd dots
        if not self._is_in_building(march_center_x, march_center_y):
            crowd_spots.append({
                'center': (march_center_x, march_center_y),
                'radius': 4,
                'intensity': 0.8,
                'num_dots': 12
            })
        
        # Place concentrated crowd dots
        for spot in crowd_spots:
            center = spot['center']
            radius = spot['radius']
            intensity = spot['intensity']
            num_dots = spot['num_dots']
            
            # Create circular pattern of concentrated dots
            for i in range(num_dots):
                angle = (i / num_dots) * 2 * math.pi
                
                # Add some randomness for natural clustering
                r = radius * (0.3 + 0.7 * random.random())
                noise_angle = angle + random.uniform(-0.3, 0.3)
                
                dot_x = int(center[0] + r * math.cos(noise_angle))
                dot_y = int(center[1] + r * math.sin(noise_angle))
                
                # Ensure dot is within bounds and not in building
                if (0 <= dot_x < self.grid_width and 
                    0 <= dot_y < self.grid_height and
                    not self._is_in_building(dot_x, dot_y)):
                    
                    # Create concentrated dot (higher intensity in small area)
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            x, y = dot_x + dx, dot_y + dy
                            if (0 <= x < self.grid_width and 
                                0 <= y < self.grid_height):
                                distance = math.sqrt(dx*dx + dy*dy)
                                dot_intensity = intensity * max(0, 1 - distance/2)
                                self.crowd_density_map[x, y] = max(
                                    self.crowd_density_map[x, y], 
                                    dot_intensity
                                )
    
    def _update_police_units(self):
        """Update police unit positions and behaviors"""
        for police in self.police_units:
            # Move towards current patrol target
            target = police.patrol_route[police.current_target]
            dx = target[0] - police.x
            dy = target[1] - police.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < 2.0:  # Reached target
                police.current_target = (police.current_target + 1) % len(police.patrol_route)
            else:
                # Move towards target
                police.x += (dx / distance) * police.speed
                police.y += (dy / distance) * police.speed
            
            police.strategy_timer += 1
            
            # Occasional tear gas deployment
            if police.strategy_timer > 50 and random.random() < 0.02:
                self._deploy_tear_gas(police.x, police.y)
                police.strategy_timer = 0
            
            # Water cannon deployment near crowds
            agent_distance = math.sqrt((police.x - self.agent_pos[0])**2 + 
                                     (police.y - self.agent_pos[1])**2)
            if agent_distance < 15 and random.random() < 0.01:
                self._deploy_water_cannon(police.x, police.y, self.agent_pos)
    
    def _deploy_tear_gas(self, x: float, y: float):
        """Deploy tear gas at specified location"""
        self.tear_gas_clouds.append(TearGas(
            x=x, y=y, radius=8.0, intensity=0.8, duration=100
        ))
    
    def _deploy_water_cannon(self, x: float, y: float, target_pos: np.ndarray):
        """Deploy water cannon aimed at target position"""
        angle = math.atan2(target_pos[1] - y, target_pos[0] - x)
        self.water_cannons.append(WaterCannon(
            x=x, y=y, angle=angle, range=20.0, width=math.pi/6, 
            intensity=0.9, duration=50
        ))
    
    def _update_hazards(self):
        """Update tear gas and water cannon effects"""
        # Update tear gas clouds
        self.tear_gas_clouds = [tg for tg in self.tear_gas_clouds 
                               if tg.duration > 0]
        for tg in self.tear_gas_clouds:
            tg.duration -= 1
            tg.intensity *= 0.995  # Gradual dissipation
        
        # Update water cannons
        self.water_cannons = [wc for wc in self.water_cannons 
                             if wc.duration > 0]
        for wc in self.water_cannons:
            wc.duration -= 1
    
    def _is_in_building(self, x: float, y: float) -> bool:
        """Check if position is inside a building"""
        grid_x, grid_y = int(x), int(y)
        return (grid_x, grid_y) in self.buildings
    
    def _is_in_safe_zone(self, x: float, y: float) -> bool:
        """Check if position is in a safe zone"""
        for zone in self.safe_zones:
            distance = math.sqrt((x - zone['center'][0])**2 + 
                               (y - zone['center'][1])**2)
            if distance <= zone['radius']:
                return True
        return False
    
    def _get_tear_gas_intensity(self, x: float, y: float) -> float:
        """Get tear gas intensity at position"""
        intensity = 0.0
        for tg in self.tear_gas_clouds:
            distance = math.sqrt((x - tg.x)**2 + (y - tg.y)**2)
            if distance <= tg.radius:
                intensity = max(intensity, tg.intensity * (1 - distance / tg.radius))
        return min(intensity, 1.0)
    
    def _get_water_cannon_intensity(self, x: float, y: float) -> float:
        """Get water cannon intensity at position"""
        intensity = 0.0
        for wc in self.water_cannons:
            # Check if position is in water cannon cone
            dx, dy = x - wc.x, y - wc.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= wc.range:
                angle_to_point = math.atan2(dy, dx)
                angle_diff = abs(angle_to_point - wc.angle)
                if angle_diff > math.pi:
                    angle_diff = 2*math.pi - angle_diff
                
                if angle_diff <= wc.width / 2:
                    intensity = max(intensity, wc.intensity * (1 - distance / wc.range))
        
        return min(intensity, 1.0)
    
    def _get_nearest_police_distance(self) -> Tuple[float, float, float]:
        """Get nearest police unit position and distance"""
        if not self.police_units:
            return 50.0, 50.0, 100.0  # Default if no police
        
        min_distance = float('inf')
        nearest_x, nearest_y = 50.0, 50.0
        
        for police in self.police_units:
            distance = math.sqrt((self.agent_pos[0] - police.x)**2 + 
                               (self.agent_pos[1] - police.y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_x, nearest_y = police.x, police.y
        
        return nearest_x, nearest_y, min_distance
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation vector"""
        # Normalize positions to [0, 1]
        agent_x_norm = self.agent_pos[0] / self.grid_width
        agent_y_norm = self.agent_pos[1] / self.grid_height
        
        # Nearest police information
        police_x, police_y, police_distance = self._get_nearest_police_distance()
        police_x_norm = police_x / self.grid_width
        police_y_norm = police_y / self.grid_height
        police_distance_norm = min(police_distance / 50.0, 1.0)
        
        # Current position attributes
        crowd_density = self.crowd_density_map[int(self.agent_pos[0]), 
                                              int(self.agent_pos[1])]
        tear_gas_intensity = self._get_tear_gas_intensity(
            self.agent_pos[0], self.agent_pos[1])
        water_cannon_intensity = self._get_water_cannon_intensity(
            self.agent_pos[0], self.agent_pos[1])
        in_safe_zone = 1.0 if self._is_in_safe_zone(
            self.agent_pos[0], self.agent_pos[1]) else 0.0
        
        # Strategy shift timer (normalized)
        strategy_timer_norm = min(self.steps_since_strategy_shift / 100.0, 1.0)
        
        observation = np.array([
            agent_x_norm,
            agent_y_norm,
            police_x_norm,
            police_y_norm,
            police_distance_norm,
            crowd_density,
            tear_gas_intensity,
            water_cannon_intensity,
            in_safe_zone,
            strategy_timer_norm
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, action: int) -> Tuple[float, bool]:
        """Calculate reward and check if episode is done"""
        reward = 0.0
        done = False
        
        # Base survival reward
        reward += 1.0
        
        # Distance to nearest police penalty
        _, _, police_distance = self._get_nearest_police_distance()
        reward -= 0.1 * max(0, 15 - police_distance)  # Penalty for being too close
        
        # Hazard penalties
        tear_gas_intensity = self._get_tear_gas_intensity(
            self.agent_pos[0], self.agent_pos[1])
        water_cannon_intensity = self._get_water_cannon_intensity(
            self.agent_pos[0], self.agent_pos[1])
        
        reward -= 5.0 * tear_gas_intensity
        reward -= 10.0 * water_cannon_intensity
        
        # Check if caught by police
        for police in self.police_units:
            distance = math.sqrt((self.agent_pos[0] - police.x)**2 + 
                               (self.agent_pos[1] - police.y)**2)
            if distance < 2.0:  # Caught
                reward -= 50.0
                done = True
                break
        
        if not done:
            # Check if reached exit point
            for exit_point in self.exit_points:
                distance = math.sqrt((self.agent_pos[0] - exit_point[0])**2 + 
                               (self.agent_pos[1] - exit_point[1])**2)
                if distance < 3.0:
                    reward += 10.0
                    done = True
                    break
        
        # Episode timeout
        if self.current_step >= self.max_episode_steps:
            done = True
        
        return reward, done
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset agent position (spawn in safe area)
        safe_spawn_points = [
            (10, 10), (10, 90), (90, 10), (90, 90), (50, 10)
        ]
        spawn_point = random.choice(safe_spawn_points)
        self.agent_pos = np.array(spawn_point, dtype=np.float32)
        self.last_direction = np.array([1.0, 0.0])
        
        # Reset counters
        self.current_step = 0
        self.steps_since_strategy_shift = 0

        # Reset visited map
        self.visited_map.fill(0.0)
        
        # # Reset start position
        self.start_pos = self.agent_pos.copy()

        # Respawn dynamic elements
        self._spawn_police_units()
        self.tear_gas_clouds.clear()
        self.water_cannons.clear()
        self._update_crowd_density()
        
        observation = self._get_observation()
        info = {"episode": "started"}
        
        return observation, info
    
    def step(self, action: int):
        """Execute one step in the environment"""
        self.current_step += 1
        self.steps_since_strategy_shift += 1

        old_pos = self.agent_pos.copy()
        
        if action == 0:  # Move North
            new_pos = self.agent_pos + np.array([0, self.step_size])
            self.last_direction = np.array([0, 1])
        elif action == 1:  # Move South
            new_pos = self.agent_pos + np.array([0, -self.step_size])
            self.last_direction = np.array([0, -1])
        elif action == 2:  # Move East
            new_pos = self.agent_pos + np.array([self.step_size, 0])
            self.last_direction = np.array([1, 0])
        elif action == 3:  # Move West
            new_pos = self.agent_pos + np.array([-self.step_size, 0])
            self.last_direction = np.array([-1, 0])
        elif action == 4:  # Stay
            new_pos = self.agent_pos.copy()
        elif action == 5:  # Sprint
            new_pos = self.agent_pos + self.last_direction * self.step_size * self.sprint_multiplier
        else:
            new_pos = self.agent_pos.copy()
        
        # Boundary checks
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_height - 1)
        
        # Building collision check - ENFORCE COLLISION DETECTION
        if not self._is_in_building(new_pos[0], new_pos[1]):
            self.agent_pos = new_pos
        # If collision detected, agent stays at old position (no movement)
        
        # Update environment
        self._update_police_units()
        self._update_hazards()
        self._update_crowd_density()

        # Initialize reward
        reward = 0.0
        cell_x, cell_y = int(self.agent_pos[0]), int(self.agent_pos[1])

        # Exploration bonus
        dist_from_start = np.linalg.norm(self.agent_pos - self.start_pos)
        explore_bonus = 0.3 * (dist_from_start / max(self.grid_width, self.grid_height))
        reward += explore_bonus
        
        # New cell discovery bonus
        if self.visited_map[cell_x, cell_y] == 0:
            # Scale bonus by distance from start and crowd density
            dist_bonus = 0.5 * (dist_from_start / max(self.grid_width, self.grid_height))
            crowd_bonus = 0.3 * self.crowd_density_map[cell_x, cell_y]
            reward += 1.0 + dist_bonus + crowd_bonus  # Base + scaled bonuses
        self.visited_map[cell_x, cell_y] += 1  # Mark cell as visited

        # Movement reward/penalty
        if np.allclose(self.agent_pos, old_pos):
            # Reward movement proportional to crowd density
            crowd = self.crowd_density_map[cell_x, cell_y]
            reward += 0.2 + 0.3 * crowd

            # HAZARD AVOIDANCE BONUS MOVED HERE
            hazard_direction = np.array([0.0, 0.0])
            for wc in self.water_cannons:
                direction = self.agent_pos - np.array([wc.x, wc.y])
                if np.linalg.norm(direction) > 0:
                    hazard_direction += direction / np.linalg.norm(direction)
            
            if np.linalg.norm(hazard_direction) > 0:
                move_direction = self.agent_pos - old_pos
                alignment = np.dot(
                    move_direction / np.linalg.norm(move_direction),
                    hazard_direction / np.linalg.norm(hazard_direction)
                )
                reward += 0.3 * max(0, alignment)  # Reward hazard-avoidance
        elif action == 4:  # Stay action
            reward -= 0.5
        
        # Check if agent is near the edge of the grid
        # Penalize if too close to edges
        edge_threshold = 0.15
        x_norm = self.agent_pos[0] / self.grid_width
        y_norm = self.agent_pos[1] / self.grid_height
        edge_factor = max(
            min(x_norm, 1-x_norm),
            min(y_norm, 1-y_norm)
        )
        # quadratic penalty for being too close to edges
        if edge_factor < edge_threshold:
            penalty_strength = 2.0 * (1 - (edge_factor / edge_threshold))**2
            reward -= penalty_strength
        
        # Safe zone handling
        current_safe_zone = None
        for zone in self.safe_zones:
            if self._is_in_safe_zone(self.agent_pos[0], self.agent_pos[1]):
                current_safe_zone = zone['center']
                break

        if current_safe_zone:
            if current_safe_zone == self.last_safe_zone:
                self.safe_zone_timer += 1
                # Exponential penalty after 5 steps in same zone
                if self.safe_zone_timer > 5:
                    reward -= 0.5 * (self.safe_zone_timer - 5)
            else:
                self.safe_zone_timer = 1  # Reset timer for new zone
                reward += 3.0
            self.last_safe_zone = current_safe_zone
        else:
            self.safe_zone_timer = 0
            self.last_safe_zone = None

        # calculate hazard penalties and terminal rewards
        hazard_reward, done = self._calculate_reward()
        reward += hazard_reward

        # Calculate reward and check termination
        reward, done = self._calculate_reward(action)
        
        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "agent_pos": self.agent_pos.copy(),
            "police_count": len(self.police_units),
            "tear_gas_count": len(self.tear_gas_clouds),
            "water_cannon_count": len(self.water_cannons)
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, done, False, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Nairobi CBD Protest Navigation")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw roads
        for road_pos in self.roads:
            pygame.draw.rect(
                self.screen, self.colors['road'],
                (road_pos[0] * self.cell_size, road_pos[1] * self.cell_size,
                 self.cell_size, self.cell_size)
            )
        
        # Draw buildings
        for building_pos in self.buildings:
            pygame.draw.rect(
                self.screen, self.colors['building'],
                (building_pos[0] * self.cell_size, building_pos[1] * self.cell_size,
                 self.cell_size, self.cell_size)
            )
        
        # Draw crowd density (heat map)
        crowd_surface = pygame.Surface((self.screen_width, self.screen_height))
        crowd_surface.set_alpha(100)
        for x in range(0, self.grid_width, 2):
            for y in range(0, self.grid_height, 2):
                density = self.crowd_density_map[x, y]
                if density > 0.3:
                    color_intensity = int(255 * min(density, 1.0))
                    color = (color_intensity, 255 - color_intensity, 0)
                    pygame.draw.rect(
                        crowd_surface, color,
                        (x * self.cell_size, y * self.cell_size,
                         self.cell_size * 2, self.cell_size * 2)
                    )
        self.screen.blit(crowd_surface, (0, 0))
        
        # Draw safe zones
        safe_surface = pygame.Surface((self.screen_width, self.screen_height))
        safe_surface.set_alpha(100)
        for zone in self.safe_zones:
            pygame.draw.circle(
                safe_surface, self.colors['safe_zone'],
                (int(zone['center'][0] * self.cell_size),
                 int(zone['center'][1] * self.cell_size)),
                int(zone['radius'] * self.cell_size)
            )
        self.screen.blit(safe_surface, (0, 0))
        
        # Draw tear gas clouds
        gas_surface = pygame.Surface((self.screen_width, self.screen_height))
        gas_surface.set_alpha(128)
        for tg in self.tear_gas_clouds:
            pygame.draw.circle(
                gas_surface, self.colors['tear_gas'],
                (int(tg.x * self.cell_size), int(tg.y * self.cell_size)),
                int(tg.radius * self.cell_size)
            )
        self.screen.blit(gas_surface, (0, 0))
        
        # Draw water cannons
        cannon_surface = pygame.Surface((self.screen_width, self.screen_height))
        cannon_surface.set_alpha(128)
        for wc in self.water_cannons:
            # Draw cone for water cannon
            points = []
            center = (int(wc.x * self.cell_size), int(wc.y * self.cell_size))
            points.append(center)
            
            for angle_offset in np.linspace(-wc.width/2, wc.width/2, 10):
                angle = wc.angle + angle_offset
                end_x = center[0] + int(wc.range * self.cell_size * math.cos(angle))
                end_y = center[1] + int(wc.range * self.cell_size * math.sin(angle))
                points.append((end_x, end_y))
            
            if len(points) > 2:
                pygame.draw.polygon(cannon_surface, self.colors['water_cannon'], points)
        self.screen.blit(cannon_surface, (0, 0))
        
        # Draw police units
        for police in self.police_units:
            pygame.draw.circle(
                self.screen, self.colors['police'],
                (int(police.x * self.cell_size), int(police.y * self.cell_size)),
                self.cell_size
            )
        
        # Draw agent
        pygame.draw.circle(
            self.screen, self.colors['agent'],
            (int(self.agent_pos[0] * self.cell_size), 
             int(self.agent_pos[1] * self.cell_size)),
            self.cell_size // 2
        )
        
        # Draw exit points
        for exit_point in self.exit_points:
            pygame.draw.rect(
                self.screen, (0, 255, 255),
                (exit_point[0] * self.cell_size - self.cell_size,
                 exit_point[1] * self.cell_size - self.cell_size,
                 self.cell_size * 2, self.cell_size * 2), 3
            )
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
    
    def close(self):
        """Clean up rendering resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()