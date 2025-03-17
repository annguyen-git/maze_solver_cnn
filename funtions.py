import os
import random
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from collections import deque
from tqdm import tqdm
from PIL import Image


# Use BFS to check if exsit solution
def is_path_exists(grid, start, goal):
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([start])
    visited[start] = True
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            return True
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                not visited[new_row, new_col] and grid[new_row, new_col] != 1):
                queue.append((new_row, new_col))
                visited[new_row, new_col] = True
    return False


# Generate mazes
def generate_grid(size=9, wall_prob=0.3, max_attempts=100):
    random_seed = random.randint(0, 10000)
    rng = random.Random(random_seed)
    
    for _ in range(max_attempts):
        grid = np.zeros((size, size))
        for row in range(size):
            for col in range(size):
                if (row == size-1 and col == 0) or (row == 0 and col == size-1):
                    continue
                if rng.random() < wall_prob:
                    grid[row, col] = 1
        
        if is_path_exists(grid, (size-1, 0), (0, size-1)):
            grid[size-1, 0] = 2  # Start
            grid[0, size-1] = 3  # Goal
            return grid
    
    # Fallback: simple maze with guaranteed path
    grid = np.ones((size, size))
    for i in range(size-1, -1, -1):
        grid[i, 0] = 0
    for j in range(1, size):
        grid[0, j] = 0
    grid[size-1, 0] = 2
    grid[0, size-1] = 3
    return grid


# Find path in mazes
def find_path(grid):
    rows, cols = grid.shape
    start = (rows-1, 0)
    goal = (0, cols-1) 
    
    parent = {}
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([start])
    visited[start] = True
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path
        
        row, col = current
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                not visited[new_row, new_col] and grid[new_row, new_col] != 1):
                queue.append((new_row, new_col))
                visited[new_row, new_col] = True
                parent[(new_row, new_col)] = current
    return []


# Save to file
def save_maze_image(grid, path=None, output_path=None, dpi=100):
    plt.figure(figsize=(5, 5))
    
    if path:
        grid_copy = grid.copy()
        for row, col in path:
            if (row, col) != (grid.shape[0]-1, 0) and (row, col) != (0, grid.shape[1]-1):
                grid_copy[row, col] = 4  # Path
        colors = ['white', 'gray', 'red', 'green', 'yellow']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        plt.pcolormesh(grid_copy, cmap=cmap, edgecolors='none')
    else:
        # Original maze without solution
        colors = ['white', 'gray', 'red', 'green']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        plt.pcolormesh(grid, cmap=cmap, edgecolors='none')
    
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error saving file: {e}")
        plt.close()
        return None


##### Main Function 1 #####
def generate_maze_dataset(num_mazes=1000, size=9, wall_prob=0.3, output_dir="maze_dataset", save_format="png", dpi=100):
    os.makedirs(os.path.join(output_dir, "mazes"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "solutions"), exist_ok=True)
    
    for i in tqdm(range(num_mazes), desc="Generating mazes"):
        try:
            # Generate maze with path
            grid = generate_grid(size=size, wall_prob=wall_prob)
            path = find_path(grid)
            
            if not path:
                continue
                
            # Save maze image
            maze_path = os.path.join(output_dir, "mazes", f"maze_{i}.{save_format}")
            save_maze_image(grid, path=None, output_path=maze_path, dpi=dpi)
            
            # Save solution image
            solution_path = os.path.join(output_dir, "solutions", f"solution_{i}.{save_format}")
            save_maze_image(grid, path=path, output_path=solution_path, dpi=dpi)
            
        except Exception as e:
            print(f"Error generating maze {i}: {e}")
    
    print(f"Successfully generated {num_mazes} maze images in {output_dir}")


# Display some example
def example_mazes(number_of_mazes, maze_dir="maze_dataset/mazes", solution_dir="maze_dataset/solutions"):
    maze_files = [f for f in os.listdir(maze_dir) if f.startswith("maze_")]
    
    selected_files = random.sample(maze_files, min(number_of_mazes, len(maze_files)))
    
    # Display each selected maze and its corresponding solution
    for maze_file in selected_files:
        solution_file = maze_file.replace("maze_", "solution_")
        
        maze_image = Image.open(os.path.join(maze_dir, maze_file))
        solution_image = Image.open(os.path.join(solution_dir, solution_file))
        
        # Plot the maze and solution side by side
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(maze_image)
        axes[0].set_title("Maze")
        axes[0].axis('off')
        
        axes[1].imshow(solution_image)
        axes[1].set_title("Solution")
        axes[1].axis('off')
        
        plt.show()


# Clean up
def clean_maze_dataset(base_dir="maze_dataset"):
    subdirs = ["mazes", "solutions","grids"]
    files_removed = False  # Flag to track if any files were removed
    
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        
        # Check if the subdirectory exists
        if os.path.exists(dir_path):
            # Iterate over all files in the subdirectory
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                # Remove the file
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_removed = True  # Set flag to True if a file is removed
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
        else:
            print(f"Directory {dir_path} does not exist.")
    
    if files_removed:
        print("All files in the maze_dataset have been successfully removed.")
    else:
        print("No files were found to remove.")




