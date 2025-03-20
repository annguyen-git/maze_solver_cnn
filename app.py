import streamlit as st
import os
import cnn_rl
import sys
import io
import time
from PIL import Image

# Function to capture printed output from solve_maze
def capture_output(func, *args, **kwargs):
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    func(*args, **kwargs)
    sys.stdout = sys.__stdout__ 
    return output_buffer.getvalue()

# Function to process and solve the maze
def process_maze(uploaded_file, training_episodes, maze_size):

    if uploaded_file is not None:
        maze_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(maze_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(maze_path, caption="Uploaded Maze", use_column_width=True)
        
        start_time = time.time()
        
        with st.spinner(f"Solving the maze with {training_episodes} training episodes and size {maze_size}... Please wait ⏳"):
            output_text = capture_output(cnn_rl.solve_maze, maze_path, training_episodes=training_episodes, maze_size=maze_size)
        
        end_time = time.time()
        time_taken = end_time - start_time
        
 
        st.text_area("Solver Output", output_text, height=200)
        solution_gif = "maze_solution.gif"
        if os.path.exists(solution_gif):
            st.image(solution_gif, caption="Solved Maze Animation", use_column_width=True)
        else:
            st.error("Error: The solution file maze_solution.gif was not found.")
        
        solution_img = "final_solution_path.png"
        if os.path.exists(solution_img):
            st.image(solution_img, caption="Final Solved Maze", use_column_width=True)
        else:
            st.error("Error: The solution image maze_solution.png was not found.")
        
        # Display the time taken
        st.info(f"Time Taken: {time_taken:.2f} seconds")
        
        st.success("Maze Solved!")

# Streamlit UI
st.title("Maze Solver with CNN + RL")
uploaded_file = st.file_uploader("Upload a maze image", type=["png", "jpg", "jpeg"])
training_episodes = st.text_input("Enter number of training episodes", value="10")
maze_size = st.text_input("Choose maze size", value="9")

valid_inputs = True

if training_episodes.isdigit():
    training_episodes = int(training_episodes)
else:
    st.error("Please enter a valid number for training episodes.")
    valid_inputs = False

if maze_size.isdigit():
    maze_size = int(maze_size)
else:
    st.error("Please enter a valid number for maze size.")
    valid_inputs = False

if uploaded_file is not None and valid_inputs:
    if st.button("Solve Maze"):
        process_maze(uploaded_file, training_episodes, maze_size)