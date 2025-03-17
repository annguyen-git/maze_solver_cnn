import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, BatchNormalization

def process_and_train_no_encode(mazes_dir, solutions_dir, epochs=10, batch_size=8, num_samples=2):
    def load_image_data(mazes_dir, solutions_dir):
        maze_files = [f for f in os.listdir(mazes_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        X, y = [], []
        
        for maze_file in maze_files:
            maze_path = os.path.join(mazes_dir, maze_file)
            solution_file = maze_file.replace("maze_", "solution_")
            solution_path = os.path.join(solutions_dir, solution_file)
            
            if not os.path.exists(maze_path) or not os.path.exists(solution_path):
                continue
            
            maze_img = cv2.imread(maze_path, cv2.IMREAD_COLOR)
            solution_img = cv2.imread(solution_path, cv2.IMREAD_COLOR)
            maze_img = cv2.resize(maze_img, (27, 27))
            solution_img = cv2.resize(solution_img, (27, 27))
            
            X.append(maze_img)
            y.append(solution_img)
        
        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.float32) / 255.0
        
        return X, y

#     def build_deeper_cnn(input_shape):
#         model = Sequential([
#             Input(shape=input_shape),
#             Conv2D(32, (3, 3), activation='relu', padding='same'),
#             MaxPooling2D((2, 2)),
#             Conv2D(64, (3, 3), activation='relu', padding='same'),
#             MaxPooling2D((2, 2)),
#             Conv2D(128, (3, 3), activation='relu', padding='same'),
#             UpSampling2D((2, 2)),
#             Conv2D(64, (3, 3), activation='relu', padding='same'),
#             UpSampling2D((2, 2)),
#             Conv2D(32, (3, 3), activation='relu', padding='same'),
#             Conv2D(3, (3, 3), activation='sigmoid', padding='same')
#         ])
        
#         model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
        
#         #model.save('cnn_no_encode.keras')
#         #print("Model saved as cnn_no_encode.keras")

#        return model
    
    def build_deeper_cnn(input_shape):
        model = Sequential([
            Input(shape=input_shape),
            
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Add padding='same' to the final layer to maintain dimensions
            Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Maze solver model created with BatchNorm and Dropout")
        return model

    X, y = load_image_data(mazes_dir, solutions_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_shape = X.shape[1:]
    model = build_deeper_cnn(input_shape)
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    predictions = model.predict(X_test)
    
    indices = np.random.choice(len(y_test), num_samples, replace=False)
    
    for idx in indices:
        plt.figure(figsize=(6, 3))
        
        plt.subplot(1, 2, 1)
        plt.imshow(y_test[idx])
        plt.title('True Solution')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(predictions[idx])
        plt.title('Predicted Solution')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return model



def evaluate_on_new_data_no_encode(model, test_mazes_dir):
    test_files = [f for f in os.listdir(test_mazes_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for test_file in test_files:
        test_path = os.path.join(test_mazes_dir, test_file)
        
        if not os.path.exists(test_path):
            continue
        
        test_img = cv2.imread(test_path, cv2.IMREAD_COLOR)
        test_img_resized = cv2.resize(test_img, (128, 128))
        test_img_normalized = np.array(test_img_resized, dtype=np.float32) / 255.0
        test_img_normalized = np.expand_dims(test_img_normalized, axis=0)  # Add batch dimension
        
        prediction = model.predict(test_img_normalized)
        
        plt.figure(figsize=(6, 3))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
        plt.title(f'Original Maze: {test_file}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(prediction[0])
        plt.title('Predicted Solution')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()



