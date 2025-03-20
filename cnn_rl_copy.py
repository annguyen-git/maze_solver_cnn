import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque
import os
import cv2
from PIL import Image

# Define maze environment
class MazeEnvironment:
    def __init__(self, maze_img_path, maze_size=9):
        self.maze_img_path = maze_img_path
        self.maze_size = maze_size  # Store maze_size as an instance variable
        
        self.maze_img = cv2.imread(maze_img_path)
        if self.maze_img is None:
            raise ValueError(f"Could not load maze image from {maze_img_path}")
            
        # colors mapping
        self.COLOR_MAP = {
            'red': 1,      # Start
            'green': 2,    # Goal
            'yellow': 3,   # path - chose
            'white': 0,    # blank - potential path
            'gray': 4      # Wall
        }
        
        # Process the maze image to create grid
        self.grid = self._process_maze_image(maze_size)
        self.height, self.width = maze_size, maze_size
        
        # Find start and end positions
        self.start_pos = self._find_position(self.COLOR_MAP['red'])
        self.end_pos = self._find_position(self.COLOR_MAP['green'])
        
        # use default positions if no start or end
        if self.start_pos is None:
            self.start_pos = (0, 0)
        if self.end_pos is None:
            self.end_pos = (self.height-1, self.width-1)
            
        self.current_pos = self.start_pos
        self.visited = set()
        self.path_history = []
        
    def _process_maze_image(self, maze_size):
        """Process the maze image using color classification to create a grid"""
        img_height, img_width = self.maze_img.shape[:2]
        cell_height, cell_width = img_height // maze_size, img_width // maze_size
        
        grid = np.zeros((maze_size, maze_size), dtype=np.int32)
        
        # Process each cell
        for i in range(maze_size):
            for j in range(maze_size):
                # Get center of cell
                y_center = i * cell_height + cell_height // 2
                x_center = j * cell_width + cell_width // 2
                
                # Get RGB values (note: OpenCV uses BGR)
                b, g, r = self.maze_img[y_center, x_center]
                
                # Color classification based on provided thresholds
                if r > 200 and g < 100 and b < 100:  # Red - start
                    grid[i, j] = self.COLOR_MAP['red']
                elif g > 120 and r < 60 and b < 50:  # Green - goal
                    grid[i, j] = self.COLOR_MAP['green']
                elif r > 200 and g > 200 and b < 100:  # Yellow
                    grid[i, j] = self.COLOR_MAP['yellow']
                elif r > 200 and g > 200 and b > 200:  # White - path
                    grid[i, j] = self.COLOR_MAP['white']
                elif 100 < r < 150 and 100 < g < 150 and 100 < b < 150:  # Gray - walls
                    grid[i, j] = self.COLOR_MAP['gray']
                else:
                    # Default to wall
                    grid[i, j] = self.COLOR_MAP['gray']
        
        return grid
        
    def _find_position(self, value):
        """Find position of a specific value in the grid"""
        positions = np.where(self.grid == value)
        if len(positions[0]) > 0:
            return (positions[0][0], positions[1][0])
        return None
    
    def reset(self):
        """Reset the environment to initial state and return observation"""
        self.current_pos = self.start_pos
        self.visited = set([self.start_pos])
        self.path_history = [self.start_pos]
        state = self._get_state(self.maze_size)  # Use instance variable
        return state
    
    def _get_state(self, maze_size):
        """Get the current state representation"""
        # Create a 5x5 window centered on current position
        state = np.zeros((5, 5), dtype=np.float32)
        y, x = self.current_pos
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                ny, nx = y + i, x + j
                if 0 <= ny < maze_size and 0 <= nx < maze_size:
                    # 1 for walls, 0 for paths
                    state[i+2, j+2] = 1 if self.grid[ny, nx] == self.COLOR_MAP['gray'] else 0
                else:
                    state[i+2, j+2] = 1  # wall if outside maze
        
        # Add end position information relative to current position
        end_y, end_x = self.end_pos
        state = np.stack([
            state,
            np.ones((5, 5)) * (y - end_y) / maze_size,
            np.ones((5, 5)) * (x - end_x) / maze_size
        ], axis=-1)
        
        return state
    
    def step(self, action):
        """Take a step based on the action and return the new state, reward, done flag"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        y, x = self.current_pos
        new_y, new_x = y + dy, x + dx
        
        # check if new position is valid
        if not (0 <= new_y < self.maze_size and 0 <= new_x < self.maze_size):
            return self._get_state(self.maze_size), -1, False, {}
        
        # check if hitting a wall
        if self.grid[new_y, new_x] == self.COLOR_MAP['gray']:
            return self._get_state(self.maze_size), -1, False, {}
        
        # move to new position
        self.current_pos = (new_y, new_x)
        self.path_history.append(self.current_pos)
        
        # check if reached the end
        if self.current_pos == self.end_pos:
            # Higher reward for shorter paths
            return self._get_state(self.maze_size), 10 + (100 - len(self.path_history)), True, {}
        
        # stronger penalty for revisiting cells to encourage exploration of new paths
        if self.current_pos in self.visited:
            reward = -0.5
        else:
            # Calculate reward based on distance to goal
            end_y, end_x = self.end_pos
            prev_dist = abs(y - end_y) + abs(x - end_x)
            new_dist = abs(new_y - end_y) + abs(new_x - end_x)
            reward = 0.5 if new_dist < prev_dist else -0.2
        
        self.visited.add(self.current_pos)
        return self._get_state(self.maze_size), reward, False, {}
    
    def render(self, mode='rgb_array'):
        """Render the current state of the environment"""
        # Create a visualization of the maze
        img = np.copy(self.maze_img)
        cell_h, cell_w = img.shape[0] // self.maze_size, img.shape[1] // self.maze_size
        
        # Colors for rendering
        RED = (0, 0, 255)      # Start
        GREEN = (0, 255, 0)    # End
        BLUE = (255, 0, 0)     # Current position
        YELLOW = (0, 255, 255) # Path
        
        # Draw path
        for i in range(1, len(self.path_history)-1):
            y, x = self.path_history[i]
            cv2.rectangle(img, (x*cell_w, y*cell_h), ((x+1)*cell_w, (y+1)*cell_h), YELLOW, -1)
        
        # Draw current position
        y, x = self.current_pos
        cv2.rectangle(img, (x*cell_w, y*cell_h), ((x+1)*cell_w, (y+1)*cell_h), BLUE, -1)
        
        # Draw start and end (if not already visible in the original image)
        sy, sx = self.start_pos
        ey, ex = self.end_pos
        
        # Add grid lines for clarity
        for i in range(1, self.maze_size):
            cv2.line(img, (0, i*cell_h), (img.shape[1], i*cell_h), (150, 150, 150), 1)
            cv2.line(img, (i*cell_w, 0), (i*cell_w, img.shape[0]), (150, 150, 150), 1)
        
        return img

# CNN model for Q-learning
def create_model():
    input_layer = tf.keras.layers.Input(shape=(5, 5, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(4)(x)  # 4 actions: up, right, down, left
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        self.memory = deque(maxlen=10000)  # Increase memory size
        self.gamma = 0.99  # Increase future reward importance
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Slower decay for more exploration
        self.batch_size = 64  # Larger batch size
        self.target_update_counter = 0
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        target_vals = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_vals[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
# Function to train the agent
def train_agent(env, agent, episodes=100):
    frames = []
    best_path = None
    best_steps = float('inf')
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            
            # Save frame every 5 steps
            if steps % 5 == 0:
                frame = env.render()
                frame = cv2.putText(frame, f'Episode: {episode+1}', (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2, cv2.LINE_AA)
                frames.append(frame)
                
            # Update target network every 10 steps
            if steps % 10 == 0:
                agent.update_target_model()
                
            agent.replay()
            
        # Check if we found a solution and if it's better than our previous best
        if done and steps < best_steps:
            best_steps = steps
            best_path = env.path_history.copy()
            best_frames = [env.render()]
            print(f"New best path found! Steps: {best_steps}")
            
        print(f"Episode: {episode+1}/{episodes}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")
        
    # visualize the best path
    if best_path:
        # Reconstruct the best path
        env.reset()
        env.path_history = best_path
        final_frame = env.render()
        frames.append(final_frame)
        print(f"Best solution found: {best_steps} steps")
            
    return frames, best_path

# Save frames as GIF
def save_frames_as_gif(frames, filename='maze_solution.gif'):
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_frames[0].save(
        filename,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )
    print(f"GIF saved as {filename}")
    
# Main function
def solve_maze(maze_img_path, training_episodes=10, maze_size=9):
    env = MazeEnvironment(maze_img_path, maze_size)
    
    model = create_model()
    model.summary()
    target_model = create_model()
    target_model.set_weights(model.get_weights())
    
    agent = DQNAgent(model, target_model)
    
    frames, best_path = train_agent(env, agent, episodes=training_episodes)

    save_frames_as_gif(frames)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))
    if best_path:
        plt.title(f"Optimal Maze Solution ({len(best_path)-1} steps)")
    plt.axis('off')
    
    plt.savefig('final_solution_path.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    return frames, best_path