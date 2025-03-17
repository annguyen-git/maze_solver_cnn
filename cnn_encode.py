import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import cv2
import random

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import Dropout, BatchNormalization


#--------------------PART 1: ENCODING--------------------
# 0: white/empty, 1: gray/wall, 2: red/start, 3: green/end, 4: yellow/path
COLOR_MAP = {
    'white': 0,  # Empty cell
    'gray': 1,   # Wall
    'red': 2,    # Start
    'green': 3,  # End
    'yellow': 4  # Path
}

# Resize images and then encode to matrices
def encode_image(image_path, grid_size=9):
    img = cv2.imread(image_path)
    
    # Resize to grid_size x grid_size (one pixel per maze cell)
    resized_img = cv2.resize(img, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    
    # Create empty grid for numerical representation
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    
    # Process each pixel/cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Get BGR color of pixel
            bgr_color = resized_img[i, j]
            
            # Map to numerical value based on dominant color
            b, g, r = bgr_color
            
            # Simple color classification - I do this manually, so please adjust thresholds as needed
            if r > 200 and g < 100 and b < 100:  # Red
                grid[i, j] = COLOR_MAP['red']
            elif g > 120 and r < 60 and b < 50:  # Green
                grid[i, j] = COLOR_MAP['green']
            elif r > 200 and g > 200 and b < 100:  # Yellow
                grid[i, j] = COLOR_MAP['yellow']
            elif r > 200 and g > 200 and b > 200:  # White
                grid[i, j] = COLOR_MAP['white']
            elif 100 < r < 150 and 100 < g < 150 and 100 < b < 150:  # Gray
                grid[i, j] = COLOR_MAP['gray']
    
    return grid


# Convert solution images to grids and save them to the specified directory
def convert_and_save_solutions_to_grids(solutions_dir, grids_dir, grid_size=9):
    os.makedirs(grids_dir, exist_ok=True)
    
    solution_files = [f for f in os.listdir(solutions_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Converting {len(solution_files)} solution images to grids...")
    
    # Process each solution image
    for solution_file in tqdm(solution_files):
        # Get full path to solution
        solution_path = os.path.join(solutions_dir, solution_file)
        
        grid = encode_image(solution_path, grid_size)
        
        # ssve grid to output directory using the same filename but with .npy extension
        grid_filename = os.path.splitext(solution_file)[0] + '.npy'
        grid_path = os.path.join(grids_dir, grid_filename)
        np.save(grid_path, grid)
    
    print(f"Conversion complete. Grids saved to {grids_dir}")


# Get just some images
def display_sample_mazes(solutions_dir, grids_dir, num_samples=2):
    solution_files = [f for f in os.listdir(solutions_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    samples = random.sample(solution_files, min(num_samples, len(solution_files)))
    
    for sample in samples:
        base_name = os.path.splitext(sample)[0]
        solution_path = os.path.join(solutions_dir, sample)
        grid_path = os.path.join(grids_dir, base_name + '.npy')
        
        if not os.path.exists(grid_path):
            print(f"Grid file not found: {grid_path}")
            continue
        
        solution_img = cv2.cvtColor(cv2.imread(solution_path), cv2.COLOR_BGR2RGB)
        grid = np.load(grid_path)
        
        plt.figure(figsize=(6, 3))
        
        plt.subplot(1, 2, 1)
        plt.imshow(solution_img)
        plt.title("Solution")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(grid, cmap='gray')
        plt.title("Grid Representation")
        plt.axis('off')
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                text_color = 'white' if value < 2 else 'black'
                plt.text(j, i, str(value), ha='center', va='center', color=text_color)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Displayed {base_name}")

# Multichanel split
def load_dataset_multichannel(grids_dir, num_grids=None):
    print("Loading dataset from:", grids_dir)
    
    grid_files = [f for f in os.listdir(grids_dir) if f.endswith('.npy')]
    print(f"Found {len(grid_files)} grid files")

    print(f"Using {num_grids} from grid files")

    if len(grid_files) == 0:
        raise ValueError(f"No grid files found in {grids_dir}")
    
    if num_grids is not None:
        grid_files = grid_files[:num_grids]
    
    sample_grid = np.load(os.path.join(grids_dir, grid_files[0]))
    grid_size = sample_grid.shape[0]
    
    X = np.zeros((len(grid_files), grid_size, grid_size, 4))
    y = np.zeros((len(grid_files), grid_size, grid_size, 1))
    
    for i, file in enumerate(tqdm(grid_files, desc="Loading grids")):
        grid = np.load(os.path.join(grids_dir, file))
        
        X[i, :, :, 0] = (grid == 1).astype(np.float32)
        X[i, :, :, 1] = (grid == 2).astype(np.float32)
        X[i, :, :, 2] = (grid == 3).astype(np.float32)
        #X[i, :, :, 3] = (grid == 0).astype(np.float32)
        
        y[i, :, :, 0] = (grid == 4).astype(np.float32)
    
    return X, y, grid_files


# Display some example
def display_example_multichannel(grid_file, X, y, index=0, grids_dir=None):
    original_grid = np.load(os.path.join(grids_dir, grid_file))
    
    plt.figure(figsize=(9, 6))
    
    # Function to annotate the grid
    def annotate_grid(ax, grid):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = int(grid[i, j])
                color = 'black' if value >= 3 or value == 1 else 'white'
                ax.text(j, i, value, ha='center', va='center', color=color)
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_grid, cmap='gray')
    plt.title("Original Grid")
    plt.axis('off')
    annotate_grid(plt.gca(), original_grid)
    
    # Display each channel separately
    plt.subplot(2, 3, 2)
    plt.imshow(X[index, :, :, 0], cmap='Blues')
    plt.title("Channel 0: Walls")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(X[index, :, :, 1], cmap='Greens')
    plt.title("Channel 1: Start")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(X[index, :, :, 2], cmap='Reds')
    plt.title("Channel 2: End")
    plt.axis('off')
    
    # plt.subplot(2, 3, 5)
    # plt.imshow(X[index, :, :, 3], cmap='Purples')
    # plt.title("Channel 3: Empty")
    # plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(y[index, :, :, 0], cmap='Oranges')
    plt.title("Target: Path")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#--------------------PART 2: MODEL--------------------
# Model
# def build_cnn_model(input_shape):
#     """Build a CNN model for maze solving."""
#     model = Sequential([
#         Input(shape=input_shape),
        
#         # Feature extraction
#         Conv2D(32, (3, 3), activation='relu', padding='same'),
#         Conv2D(64, (3, 3), activation='relu', padding='same'),
#         Conv2D(128, (3, 3), activation='relu', padding='same'),
        
#         # Path prediction
#         Conv2D(64, (3, 3), activation='relu', padding='same'),
#         Conv2D(32, (3, 3), activation='relu', padding='same'),
#         Conv2D(1, (1, 1), activation='sigmoid')
#     ])
    
#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy', exact_match_accuracy]
#     )
    
#     return model


def build_cnn_model(input_shape):
    """Build a CNN model for maze solving with regularization and batch normalization."""
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),Dropout(0.3),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),Dropout(0.3),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),Dropout(0.3),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),Dropout(0.3),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),Dropout(0.3),
        
        Conv2D(1, (1, 1), activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', exact_match_accuracy]
    )
    
    return model


def exact_match_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    
    batch_size = tf.shape(y_true)[0]
    y_true_flat = tf.reshape(y_true, [batch_size, -1])
    y_pred_flat = tf.reshape(y_pred_binary, [batch_size, -1])
    
    matches = tf.reduce_all(tf.equal(y_true_flat, y_pred_flat), axis=1)
    
    return tf.reduce_mean(tf.cast(matches, tf.float32))

def calculate_exact_match_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    exact_matches = 0
    total_samples = y_true.shape[0]
    
    for i in range(total_samples):
        if np.array_equal(y_true[i], y_pred_binary[i]):
            exact_matches += 1
    
    return exact_matches / total_samples

def visualize_paths(X_test, y_test, predictions, indices=None, num_samples=2):
    if indices is None:
        indices = np.random.choice(len(y_test), min(num_samples, len(y_test)), replace=False)
    
    for idx in indices:
        plt.figure(figsize=(9, 3))
        
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[idx, :, :, 0], cmap='gray')
        plt.title('Maze (X)')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(y_test[idx, :, :, 0], cmap='gray')
        plt.title('True Path (y)')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        binary_pred = (predictions[idx, :, :, 0] > 0.5).astype(int)
        plt.imshow(binary_pred, cmap='gray')
        plt.title('Predicted Path')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def train_and_evaluate(X, y, epochs=10, batch_size=8, num_samples=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_shape = X.shape[1:]  # (grid_size, grid_size, 4)
    
    model = build_cnn_model(input_shape)
    
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    print("\nEvaluating model...")
    test_loss, test_acc, test_exact_match = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy (cell-wise): {test_acc:.4f}")
    print(f"Test exact match accuracy: {test_exact_match:.4f}")
    
    predictions = model.predict(X_test)
    
    manual_exact_match = calculate_exact_match_accuracy(y_test, predictions)
    print(f"Manual exact match calculation: {manual_exact_match:.4f}")
    
    print("\nVisualizing sample predictions...")
    visualize_paths(X_test, y_test, predictions, num_samples=num_samples)
    
    return model, history, X_test, y_test, predictions

def plot_training_history(history):
    plt.figure(figsize=(9, 3))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['exact_match_accuracy'], label='Training Exact Match')
    plt.plot(history.history['val_exact_match_accuracy'], label='Validation Exact Match')
    plt.title('Training and Validation Exact Match Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



def visualize_maze_comparison(X_sample, y_true, y_pred):
    # Create empty grids
    grid_size = X_sample.shape[0]
    original_grid = np.zeros((grid_size, grid_size), dtype=int)
    solved_grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Set walls (value 1)
    original_grid[X_sample[:, :, 0] == 1] = 1
    solved_grid[X_sample[:, :, 0] == 1] = 1
    
    # Set start point (value 2)
    original_grid[X_sample[:, :, 1] == 1] = 2
    solved_grid[X_sample[:, :, 1] == 1] = 2
    
    # Set end point (value 3)
    original_grid[X_sample[:, :, 2] == 1] = 3
    solved_grid[X_sample[:, :, 2] == 1] = 3
    
    # Empty spaces stay as 0
    
    # Add path (value 4) to solved grid only
    solved_grid[y_pred[:, :, 0] == 1] = 4
    
    # 0: white/empty, 1: gray/wall, 2: red/start, 3: green/end, 4: yellow/path
    colors = ['white', 'gray', 'red', 'green', 'yellow']
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_grid, cmap=cmap, vmin=0, vmax=4)
    plt.title('Unsolved Maze')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(solved_grid, cmap=cmap, vmin=0, vmax=4)
    plt.title('Solved Maze')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()