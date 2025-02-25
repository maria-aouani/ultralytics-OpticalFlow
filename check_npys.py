import numpy as np

# Path to your .npy file
file_path = r'C:\Users\Maria\Desktop\ultralytics-main\ultralytics-main\test_set_images\train\npy_files\forget00.npy'

# Load the .npy file
data = np.load(file_path)

# Print the shape of the array
print("Shape of the .npy file:", data.shape)