import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_hand_flow(prev_frame, curr_frame, keypoints, scale=0.2):
    hand_flow_data = []

    if keypoints is not None:
        for kpt in keypoints:
            # Преобразуем нормализованные координаты в пиксели
            h, w, _ = prev_frame.shape
            cx, cy = int(kpt[0] * w), int(kpt[1] * h)

            # Определяем область вокруг ключевой точки
            region_size = int(scale * h)  # Размер области (20% от высоты кадра)
            x1, y1 = max(0, cx - region_size), max(0, cy - region_size)
            x2, y2 = min(w, cx + region_size), min(h, cy + region_size)

            # Выделяем область ROI
            prev_roi = prev_frame[y1:y2, x1:x2]
            curr_roi = curr_frame[y1:y2, x1:x2]

            # Вычисляем оптический поток внутри ROI
            if prev_roi.size > 0 and curr_roi.size > 0:
                prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Рассчитываем усреднённый поток в области
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = np.mean(mag)
                mean_ang = np.mean(ang)

                # Сохраняем усреднённые значения
                hand_flow_data.append((mean_mag, mean_ang))

    return hand_flow_data

def convert_dataset_to_npy(image_dir, label_dir, output_dir, flow_area_ratio=0.2):
    """
    Convert the dataset to .npy format with optical flow.

    Args:
        image_dir (str): Directory containing the images.
        label_dir (str): Directory containing the labels.
        output_dir (str): Directory to save the .npy files.
        flow_area_ratio (float): Ratio of the area around keypoints to calculate optical flow.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of images and labels
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # Process each image and label
    for i in tqdm(range(len(image_files) - 1), desc="Processing images"):
        # Load current and next image
        prev_image_path = os.path.join(image_dir, image_files[i])
        next_image_path = os.path.join(image_dir, image_files[i + 1])

        prev_image = cv2.imread(prev_image_path)
        next_image = cv2.imread(next_image_path)

        # Convert images to grayscale for optical flow calculation
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow for the entire image
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Normalize optical flow
        flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))

        # Load label
        label_path = os.path.join(label_dir, label_files[i])
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Parse label (class, bbox, keypoints)
        class_id = int(lines[0].split()[0])
        bbox = list(map(float, lines[0].split()[1:5]))
        kpts = np.array([list(map(float, line.split())) for line in lines[1:]], dtype=np.float32)

        # Combine RGB and optical flow channels
        combined_image = np.zeros((prev_image.shape[0], prev_image.shape[1], 5), dtype=np.float32)
        combined_image[:, :, :3] = prev_image / 255.0  # Normalize RGB
        combined_image[:, :, 3:] = flow  # Add optical flow (2 channels)

        # Save as .npy file
        output_path = os.path.join(output_dir, os.path.splitext(image_files[i])[0] + '.npy')
        np.save(output_path, combined_image)

# Example usage
image_dir = r'C:\Users\Public\MediaPipe_Yolo_Lstm_222\MediaPipe_Yolo_Lstm_222\MP_Data\test_set_images\train\images'
label_dir = r'C:\Users\Public\MediaPipe_Yolo_Lstm_222\MediaPipe_Yolo_Lstm_222\MP_Data\test_set_images\train\labels'
output_dir = r'C:\Users\Maria\Desktop\ultralytics-main\ultralytics-main\test_set_images\train\npy_files'
convert_dataset_to_npy(image_dir, label_dir, output_dir)