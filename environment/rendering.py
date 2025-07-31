import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
from typing import List, Tuple, Optional
import random
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE

class NairobiCBD3DRenderer:
    """
    Advanced 3D renderer for the Nairobi CBD protest environment using PyOpenGL.
    Creates an immersive but performance-balanced visualization.
    """
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.camera_pos = [50, 80, 50]  # Camera position (x, y, z)
        self.camera_target = [50, 0, 50]  # Look at point
        self.camera_angle_x = -30  # Pitch
        self.camera_angle_y = 0    # Yaw
        self.zoom = 1.0
        
        # Animation parameters
        self.time = 0.0
        self.animation_speed = 0.02
        
        # Initialize pygame and OpenGL
        self._init_pygame_opengl()
        self._init_opengl_settings()
        self._setup_lighting()
        
        # Pre-generate 3D models
        self._create_building_models()
        self._create_vehicle_models()
        
    def _init_pygame_opengl(self):
        """Initialize Pygame with OpenGL context"""
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Nairobi CBD Protest - 3D View")
        
    def _init_opengl_settings(self):
        """Configure OpenGL settings for 3D rendering"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set perspective projection
        glMatrixMode(GL_PROJECTION)
        gluPerspective(60, (self.width / self.height), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Background color (black)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
    def _setup_lighting(self):
        """Configure lighting for the 3D scene"""
        # Ambient light
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        
        # Diffuse light (sun)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.6, 1.0])
        
        # Specular light
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Light position (sun overhead)
        glLightfv(GL_LIGHT0, GL_POSITION, [50, 100, 50, 1.0])
        
    def _create_building_models(self):
        """Pre-generate display lists for building models"""
        # Modern office tower
        self.building_tower = glGenLists(1)
        glNewList(self.building_tower, GL_COMPILE)
        self._draw_office_tower()
        glEndList()
        
        # Government building
        self.building_govt = glGenLists(1)
        glNewList(self.building_govt, GL_COMPILE)
        self._draw_government_building()
        glEndList()
        
        # Shopping complex
        self.building_mall = glGenLists(1)
        glNewList(self.building_mall, GL_COMPILE)
        self._draw_shopping_complex()
        glEndList()
        
    def _create_vehicle_models(self):
        """Pre-generate display lists for vehicle models"""
        # Police vehicle
        self.police_vehicle = glGenLists(1)
        glNewList(self.police_vehicle, GL_COMPILE)
        self._draw_police_vehicle()
        glEndList()
        
        # Water cannon truck
        self.water_cannon_truck = glGenLists(1)
        glNewList(self.water_cannon_truck, GL_COMPILE)
        self._draw_water_cannon_truck()
        glEndList()
        
    def _draw_cube(self, width: float, height: float, depth: float):
        """Draw a basic cube with given dimensions"""
        glBegin(GL_QUADS)
        
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-width/2, 0, depth/2)
        glVertex3f(width/2, 0, depth/2)
        glVertex3f(width/2, height, depth/2)
        glVertex3f(-width/2, height, depth/2)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-width/2, 0, -depth/2)
        glVertex3f(-width/2, height, -depth/2)
        glVertex3f(width/2, height, -depth/2)
        glVertex3f(width/2, 0, -depth/2)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-width/2, height, -depth/2)
        glVertex3f(-width/2, height, depth/2)
        glVertex3f(width/2, height, depth/2)
        glVertex3f(width/2, height, -depth/2)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-width/2, 0, -depth/2)
        glVertex3f(width/2, 0, -depth/2)
        glVertex3f(width/2, 0, depth/2)
        glVertex3f(-width/2, 0, depth/2)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(width/2, 0, -depth/2)
        glVertex3f(width/2, height, -depth/2)
        glVertex3f(width/2, height, depth/2)
        glVertex3f(width/2, 0, depth/2)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-width/2, 0, -depth/2)
        glVertex3f(-width/2, 0, depth/2)
        glVertex3f(-width/2, height, depth/2)
        glVertex3f(-width/2, height, -depth/2)
        
        glEnd()
        
    def _draw_office_tower(self):
        """Draw a modern office tower"""
        glColor3f(0.6, 0.6, 0.7)  # Grey concrete
        
        # Main tower structure
        self._draw_cube(8, 25, 8)
        
        # Windows (darker strips)
        glColor3f(0.2, 0.3, 0.5)
        for floor in range(1, 24, 3):
            glPushMatrix()
            glTranslatef(0, floor, 4.1)
            self._draw_cube(7, 2, 0.2)
            glPopMatrix()
            
            glPushMatrix()
            glTranslatef(0, floor, -4.1)
            self._draw_cube(7, 2, 0.2)
            glPopMatrix()
            
            glPushMatrix()
            glTranslatef(4.1, floor, 0)
            self._draw_cube(0.2, 2, 7)
            glPopMatrix()
            
            glPushMatrix()
            glTranslatef(-4.1, floor, 0)
            self._draw_cube(0.2, 2, 7)
            glPopMatrix()
            
    def _draw_government_building(self):
        """Draw a government building with columns"""
        glColor3f(0.8, 0.7, 0.6)  # Beige stone
        
        # Main building
        self._draw_cube(12, 8, 10)
        
        # Columns
        glColor3f(0.9, 0.8, 0.7)
        for i in range(-2, 3):
            glPushMatrix()
            glTranslatef(i * 3, 0, 5.5)
            self._draw_cube(0.8, 10, 0.8)
            glPopMatrix()
            
    def _draw_shopping_complex(self):
        """Draw a modern shopping complex"""
        glColor3f(0.7, 0.8, 0.9)  # Light blue glass
        
        # Main structure
        self._draw_cube(15, 12, 8)
        
        # Glass panels
        glColor4f(0.3, 0.5, 0.8, 0.6)
        glPushMatrix()
        glTranslatef(0, 6, 4.1)
        self._draw_cube(14, 10, 0.1)
        glPopMatrix()
        
    def _draw_police_vehicle(self):
        """Draw a police vehicle"""
        glColor3f(1.0, 1.0, 1.0)  # White base
        
        # Vehicle body
        self._draw_cube(2, 1, 4)
        
        # Police markings
        glColor3f(0.0, 0.0, 1.0)  # Blue stripe
        glPushMatrix()
        glTranslatef(0, 0.7, 0)
        self._draw_cube(2.1, 0.2, 4.1)
        glPopMatrix()
        
        # Light bar
        glColor3f(1.0, 0.0, 0.0)  # Red lights
        glPushMatrix()
        glTranslatef(0, 1.2, 0)
        self._draw_cube(1.5, 0.2, 0.3)
        glPopMatrix()
        
    def _draw_water_cannon_truck(self):
        """Draw a water cannon truck"""
        glColor3f(0.3, 0.3, 0.3)  # Dark grey
        
        # Truck body
        self._draw_cube(3, 1.5, 6)
        
        # Water cannon
        glColor3f(0.4, 0.4, 0.4)
        glPushMatrix()
        glTranslatef(0, 2, -1)
        glRotatef(30, 1, 0, 0)
        self._draw_cube(0.5, 0.5, 3)
        glPopMatrix()
        
    def _draw_ground_plane(self):
        """Draw the ground with roads and textures"""
        # Grass/dirt base
        glColor3f(0.5, 0.3, 0.1)  # Brown dirt
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(100, 0, 0)
        glVertex3f(100, 0, 100)
        glVertex3f(0, 0, 100)
        glEnd()
        
        # Roads
        glColor3f(0.4, 0.4, 0.4)
        
        # Horizontal roads
        for y in [20, 35, 50, 65, 80]:
            glBegin(GL_QUADS)
            glNormal3f(0, 1, 0)
            glVertex3f(10, 0.1, y-2)
            glVertex3f(90, 0.1, y-2)
            glVertex3f(90, 0.1, y+2)
            glVertex3f(10, 0.1, y+2)
            glEnd()
            
        # Vertical roads
        for x in [15, 30, 45, 60, 75]:
            glBegin(GL_QUADS)
            glNormal3f(0, 1, 0)
            glVertex3f(x-2, 0.1, 10)
            glVertex3f(x+2, 0.1, 10)
            glVertex3f(x+2, 0.1, 90)
            glVertex3f(x-2, 0.1, 90)
            glEnd()
            
    def _draw_safe_zones(self, safe_zones: List[dict]):
        """Draw safe zones as glowing green areas"""
        glColor4f(0.0, 1.0, 0.0, 0.3)
        
        for zone in safe_zones:
            center = zone['center']
            radius = zone['radius']
            
            # Draw circular safe zone
            glPushMatrix()
            glTranslatef(center[0], 0.2, center[1])
            
            glBegin(GL_TRIANGLE_FAN)
            glNormal3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            
            for angle in np.linspace(0, 2*math.pi, 32):
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                glVertex3f(x, 0, z)
            glVertex3f(radius, 0, 0)  # Close the circle
            glEnd()
            
            glPopMatrix()
            
    def _draw_tear_gas_cloud(self, x: float, y: float, radius: float, intensity: float):
        """Draw animated tear gas cloud"""
        glColor4f(1.0, 0.6, 0.0, intensity * 0.6)
        
        glPushMatrix()
        glTranslatef(x, 2, y)
        
        # Animated cloud using multiple spheres
        num_particles = 12
        for i in range(num_particles):
            angle = (i / num_particles) * 2 * math.pi + self.time
            offset_x = (radius * 0.3) * math.cos(angle)
            offset_z = (radius * 0.3) * math.sin(angle)
            offset_y = 2 * math.sin(self.time + i)
            
            glPushMatrix()
            glTranslatef(offset_x, offset_y, offset_z)
            
            # Use GLU sphere for smooth clouds
            quadric = gluNewQuadric()
            gluSphere(quadric, radius * 0.4, 8, 8)
            gluDeleteQuadric(quadric)
            
            glPopMatrix()
            
        glPopMatrix()
        
    def _draw_water_cannon_spray(self, x: float, y: float, angle: float, 
                                range_val: float, width: float, intensity: float):
        """Draw water cannon spray effect"""
        glColor4f(0.0, 0.7, 1.0, intensity * 0.4)
        
        glPushMatrix()
        glTranslatef(x, 1, y)
        glRotatef(math.degrees(angle), 0, 1, 0)
        
        # Draw cone-shaped water spray
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        
        num_segments = 12
        for i in range(num_segments + 1):
            segment_angle = (i / num_segments) * width - width/2
            spray_x = range_val * math.sin(segment_angle)
            spray_z = range_val * math.cos(segment_angle)
            glVertex3f(spray_x, -0.5, spray_z)
            
        glEnd()
        glPopMatrix()
        
    def _draw_crowd_density(self, crowd_map: np.ndarray, grid_width: int, grid_height: int):
        """Draw crowd density as concentrated dots instead of grid-based visualization"""
        # Find concentrated crowd areas and render as distinct people dots
        for x in range(0, grid_width, 2):  # Sample every 2nd cell for performance
            for y in range(0, grid_height, 2):
                density = crowd_map[x, y]
                if density > 0.3:  # Only show significant crowd concentrations
                    
                    # Animate crowd figures with subtle movement
                    bob_height = 0.2 * math.sin(self.time * 3 + x * 0.2 + y * 0.2)
                    
                    # Color based on crowd intensity
                    if density > 0.7:
                        glColor3f(1.0, 0.2, 0.2)  # Red for dense crowds (protests)
                    elif density > 0.5:
                        glColor3f(1.0, 0.6, 0.0)  # Orange for medium crowds
                    else:
                        glColor3f(0.8, 0.8, 0.2)  # Yellow for light crowds
                    
                    # Draw individual crowd member as small figure
                    glPushMatrix()
                    glTranslatef(x, 0.8 + bob_height, y)
                    
                    # Simple person representation - more distinct than grid
                    # Body
                    self._draw_cube(0.3, 1.2, 0.2)
                    
                    # Head
                    glPushMatrix()
                    glTranslatef(0, 1.5, 0)
                    quadric = gluNewQuadric()
                    gluSphere(quadric, 0.15, 6, 6)
                    gluDeleteQuadric(quadric)
                    glPopMatrix()
                    
                    # Add some variety - occasional signs/banners for protesters
                    if density > 0.6 and (x + y) % 7 == 0:
                        glColor3f(1.0, 1.0, 1.0)  # White banner
                        glPushMatrix()
                        glTranslatef(0, 2.2, 0)
                        glRotatef(math.degrees(math.sin(self.time + x)), 0, 0, 1)
                        self._draw_cube(0.8, 0.1, 0.02)
                        glPopMatrix()
                    
                    glPopMatrix()
                    
    def _draw_agent(self, pos: np.ndarray):
        """Draw the agent (protester) as a distinctive figure"""
        glColor3f(0.0, 0.4, 1.0)  # Blue protester
        
        glPushMatrix()
        glTranslatef(pos[0], 1, pos[1])
        
        # Body
        self._draw_cube(0.8, 1.8, 0.4)
        
        # Head
        glPushMatrix()
        glTranslatef(0, 2.2, 0)
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.3, 8, 8)
        gluDeleteQuadric(quadric)
        glPopMatrix()
        
        # Arms (animated)
        arm_swing = 0.3 * math.sin(self.time * 3)
        glColor3f(0.0, 0.3, 0.8)
        
        # Left arm
        glPushMatrix()
        glTranslatef(-0.6, 1.2, 0)
        glRotatef(math.degrees(arm_swing), 1, 0, 0)
        self._draw_cube(0.2, 1.0, 0.2)
        glPopMatrix()
        
        # Right arm  
        glPushMatrix()
        glTranslatef(0.6, 1.2, 0)
        glRotatef(math.degrees(-arm_swing), 1, 0, 0)
        self._draw_cube(0.2, 1.0, 0.2)
        glPopMatrix()
        
        glPopMatrix()
        
    def _draw_police_units(self, police_units: List):
        """Draw police units with vehicles"""
        for police in police_units:
            # Draw police vehicle
            glPushMatrix()
            glTranslatef(police.x, 0, police.y)
            glCallList(self.police_vehicle)
            glPopMatrix()
            
            # Draw police officer
            glColor3f(0.2, 0.2, 0.8)  # Dark blue uniform
            glPushMatrix()
            glTranslatef(police.x, 1, police.y + 2)
            
            # Simple police figure
            self._draw_cube(0.6, 1.6, 0.3)
            
            # Head
            glPushMatrix()
            glTranslatef(0, 2.0, 0)
            quadric = gluNewQuadric()
            gluSphere(quadric, 0.25, 6, 6)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            
            glPopMatrix()
            
    def _draw_buildings(self, buildings: List[Tuple[int, int]]):
        """Draw 3D buildings with better separation and maze-like appearance"""
        # Group buildings into distinct blocks for variety
        building_groups = {}
        
        # Group adjacent buildings together
        for bx, by in buildings:
            # Create building group key based on approximate regions
            group_key = (bx // 8, by // 8)  # Group every 8x8 area
            if group_key not in building_groups:
                building_groups[group_key] = []
            building_groups[group_key].append((bx, by))
        
        # Building types with different appearances
        building_types = [
            {'height': 12, 'color': (0.6, 0.6, 0.7), 'type': 'tower'},      # Office tower
            {'height': 6, 'color': (0.8, 0.7, 0.6), 'type': 'government'},   # Government
            {'height': 8, 'color': (0.7, 0.8, 0.9), 'type': 'mall'},        # Shopping
            {'height': 8, 'color': (0.5, 0.7, 0.5), 'type': 'residential'}, # Residential
        ]
        
        # Render each building group as a distinct structure
        for i, (group_key, building_cells) in enumerate(building_groups.items()):
            if len(building_cells) < 3:  # Skip very small groups
                continue
                
            building_type = building_types[i % len(building_types)]
            
            # Find the center and bounds of this building group
            min_x = min(cell[0] for cell in building_cells)
            max_x = max(cell[0] for cell in building_cells)
            min_y = min(cell[1] for cell in building_cells)
            max_y = max(cell[1] for cell in building_cells)
            
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x + 1
            depth = max_y - min_y + 1
            
            # Draw main building structure
            glColor3f(*building_type['color'])
            glPushMatrix()
            glTranslatef(center_x, building_type['height']/2, center_y)
            self._draw_cube(width, building_type['height'], depth)
            glPopMatrix()
            
            # Add architectural details based on building type
            if building_type['type'] == 'tower':
                # Windows for office towers
                glColor3f(0.2, 0.3, 0.5)
                for floor in range(2, building_type['height'], 4):
                    # Front and back windows
                    glPushMatrix()
                    glTranslatef(center_x, floor, center_y + depth/2 + 0.1)
                    self._draw_cube(width * 0.8, 1.5, 0.1)
                    glPopMatrix()
                    
                    glPushMatrix()
                    glTranslatef(center_x, floor, center_y - depth/2 - 0.1)
                    self._draw_cube(width * 0.8, 1.5, 0.1)
                    glPopMatrix()
                    
            elif building_type['type'] == 'government':
                # Columns for government buildings
                glColor3f(0.9, 0.8, 0.7)
                num_columns = min(4, int(width/2))
                for i in range(num_columns):
                    col_x = center_x - width/2 + (i + 0.5) * width/num_columns
                    glPushMatrix()
                    glTranslatef(col_x, building_type['height']/2, center_y + depth/2 + 0.3)
                    self._draw_cube(0.4, building_type['height'], 0.4)
                    glPopMatrix()
                    
            elif building_type['type'] == 'mall':
                # Glass facade for shopping centers
                glColor4f(0.3, 0.5, 0.8, 0.6)
                glPushMatrix()
                glTranslatef(center_x, building_type['height']/2, center_y + depth/2 + 0.1)
                self._draw_cube(width * 0.9, building_type['height'] * 0.8, 0.1)
                glPopMatrix()
                
            # Add rooftop details for visual interest
            glColor3f(0.4, 0.4, 0.4)  # Dark grey rooftop equipment
            if building_type['height'] > 10:  # Only for taller buildings
                glPushMatrix()
                glTranslatef(center_x, building_type['height'] + 1, center_y)
                self._draw_cube(2, 2, 1)  # HVAC unit
                glPopMatrix()
                
    def update_camera(self, target_pos: np.ndarray):
        """Update camera to follow the agent"""
        # target camera position relative to agent
        target_x = target_pos[0] + 15
        target_y = 35
        target_z = target_pos[1] + 15
        
        # smooth interpolation
        smoothing = 0.1
        self.camera_pos[0] += (target_x - self.camera_pos[0]) * smoothing
        self.camera_pos[1] += (target_y - self.camera_pos[1]) * smoothing
        self.camera_pos[2] += (target_z - self.camera_pos[2]) * smoothing
        
        self.camera_target[0] += (target_pos[0] - self.camera_target[0]) * smoothing
        self.camera_target[2] += (target_pos[1] - self.camera_target[2]) * smoothing
        
    def render_scene(self, env_state: dict):
        """Main rendering function"""
        self.time += self.animation_speed
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Update camera position
        if 'agent_pos' in env_state:
            self.update_camera(env_state['agent_pos'])
        
        # Set camera view
        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0, 1, 0
        )
        
        # Draw ground plane
        self._draw_ground_plane()
        
        # Draw buildings
        if 'buildings' in env_state:
            self._draw_buildings(env_state['buildings'])
        
        # Draw safe zones
        if 'safe_zones' in env_state:
            self._draw_safe_zones(env_state['safe_zones'])
        
        # Draw crowd density
        if 'crowd_density_map' in env_state:
            self._draw_crowd_density(
                env_state['crowd_density_map'],
                env_state.get('grid_width', 100),
                env_state.get('grid_height', 100)
            )
        
        # Draw police units
        if 'police_units' in env_state:
            self._draw_police_units(env_state['police_units'])
        
        # Draw tear gas clouds
        if 'tear_gas_clouds' in env_state:
            for tg in env_state['tear_gas_clouds']:
                self._draw_tear_gas_cloud(tg.x, tg.y, tg.radius, tg.intensity)
        
        # Draw water cannon sprays
        if 'water_cannons' in env_state:
            for wc in env_state['water_cannons']:
                self._draw_water_cannon_spray(
                    wc.x, wc.y, wc.angle, wc.range, wc.width, wc.intensity
                )
        
        # Draw agent
        if 'agent_pos' in env_state:
            self._draw_agent(env_state['agent_pos'])
        
        # Swap buffers
        pygame.display.flip()
        
    def handle_events(self):
        """Handle pygame events for camera control"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.camera_angle_x -= 5
                elif event.key == pygame.K_DOWN:
                    self.camera_angle_x += 5
                elif event.key == pygame.K_LEFT:
                    self.camera_angle_y -= 5
                elif event.key == pygame.K_RIGHT:
                    self.camera_angle_y += 5
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom *= 0.9
                elif event.key == pygame.K_MINUS:
                    self.zoom *= 1.1
        return True
    
    def capture_frame(self):
        """Capture the current OpenGL framebuffer as a numpy RGB image"""
        width, height = self.width, self.height
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        image = np.flipud(image)  # Flip vertically to match OpenGL's bottom-left origin
        return image[:, :, :3]  # Return RGB only, discard alpha channel
        
    def close(self):
        """Clean up resources"""
        pygame.quit()


def create_demo_visualization():
    """Create a standalone demo of the 3D environment"""
    from custom_env import NairobiCBDProtestEnv
    
    # Initialize environment and renderer
    env = NairobiCBDProtestEnv()
    renderer = NairobiCBD3DRenderer()
    
    # Reset environment to get initial state
    obs, info = env.reset()
    
    running = True
    clock = pygame.time.Clock()
    
    print("3D Nairobi CBD Protest Environment Demo")
    print("Controls: Arrow keys to adjust camera, +/- to zoom, ESC to quit")
    print("Showing agent taking random actions...")
    
    try:
        while running:
            # Handle events
            running = renderer.handle_events()
            
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                obs, info = env.reset()
            
            # Prepare state for renderer
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
            
            # Render scene
            renderer.render_scene(env_state)
            
            # Control frame rate
            clock.tick(30)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    create_demo_visualization()