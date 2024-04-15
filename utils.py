import itertools
import os
import random
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from config import *

def random_images(dataset_folder):
    random_images = []
    for person_folder in os.listdir(dataset_folder):
        person_folder_path = os.path.join(dataset_folder, person_folder)
        filenames = os.listdir(person_folder_path)
        if len(filenames) > 7:
            filenames = random.sample(filenames, 7)  # Select 7 random filenames
        for filename in filenames:
            random_images.append(os.path.join(person_folder_path, filename))
    return random_images

def show_triplets_from_dataloader(dataloader, num_triplets=3):
    for batch_idx, (anchor_imgs, positive_imgs, negative_imgs) in enumerate(dataloader):
        for i in range(len(anchor_imgs)):
            if i >= num_triplets:
                break
            
            # Display the anchor, positive, and negative images
            fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))
            
            # Anchor image
            axes[0].imshow(anchor_imgs[i].permute(1, 2, 0), cmap='gray')
            axes[0].axis('off')
            
            # Positive image
            axes[1].imshow(positive_imgs[i].permute(1, 2, 0), cmap='gray')
            axes[1].axis('off')
            
            # Negative image
            axes[2].imshow(negative_imgs[i].permute(1, 2, 0), cmap='gray')
            axes[2].axis('off')
            
            # Adjust spacing and remove unnecessary whitespace
            plt.subplots_adjust(wspace=0.05, hspace=0.1)
            plt.tight_layout()
            
            # Save the figure with a higher DPI for better quality
            plt.savefig('/home/nthumbav/Downloads/10707_final/scratch/triplet_{}.png'.format(DATASET), dpi=300)

def preprocess_image(image_pth):
    gray = image_pth.convert("L")
    img = np.array(gray)
    blur = cv2.medianBlur(img,3)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    erode = cv2.erode(blur, kernel, iterations=2)

    dilate = cv2.dilate(erode, kernel, iterations=1)

    _, binary = cv2.threshold(dilate, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)

    padding = 20  # Adjust the padding size as needed
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Make sure the coordinates are within the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Crop the image using the modified bounding box coordinates
    cropped_image = binary[y:y+h, x:x+w]

    # Add extra white space around the cropped image
    extra_space = np.zeros((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding), dtype=np.uint8) * 255
    extra_space[padding:-padding, padding:-padding] = cropped_image
    
    # Convert the numpy array back to PIL image
    resized_image = Image.fromarray(extra_space)

    return resized_image

def triplet_dataset_preparation(dataset_folder):
    image_paths = random_images(dataset_folder)
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Anchor_Path', 'Positive_Path', 'Negative_Path'])

    for person_folder in os.listdir(dataset_folder):
        person_folder_path = os.path.join(dataset_folder, person_folder)
        genuine_images = []
        forged_images = []
        for filename in os.listdir(person_folder_path):
            if 'original' in filename or '-G-' in filename:
                genuine_images.append(os.path.join(person_folder_path,filename))
            if 'forgeries' in filename or '-F-' in filename or 'forge' in filename:
                forged_images.append(os.path.join(person_folder_path,filename))
                
        additional_images = random.sample(image_paths, 20)
        forged_images.extend(additional_images)
        
        num_combinations = min(len(genuine_images) * (len(genuine_images) - 1) // 2, len(genuine_images) * len(forged_images))
        genuine_combinations = random.sample(list(itertools.combinations(genuine_images, 2)), num_combinations)
        forged_combinations = random.sample(list(itertools.product(genuine_images, forged_images)), num_combinations)

        # Create a DataFrame with the balanced triplets
        data = []
        for (image_1, image_2), (genuine_image, forged_image) in zip(genuine_combinations, forged_combinations):
            anchor_path = os.path.join(image_1)
            positive_path = os.path.join(image_2)
            negative_path = os.path.join(forged_image)
            data.append([anchor_path, positive_path, negative_path])

        df = df.append(pd.DataFrame(data, columns=['Anchor_Path', 'Positive_Path', 'Negative_Path']), ignore_index=True)
    return df

# function to create duplets for logisitic regression training
def duplet_dataset_preparation(dataset_folder):
    image_paths = random_images(dataset_folder)
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Image1', 'Image2', 'Label'])

    for person_folder in os.listdir(dataset_folder):

        person_folder_path = os.path.join(dataset_folder, person_folder)

        genuine_images = []
        forged_images = []
        for filename in os.listdir(person_folder_path):
            if 'original' in filename or '-G-' in filename:
                genuine_images.append(os.path.join(person_folder_path,filename))
            if 'forgeries' in filename or '-F-' in filename or 'forge' in filename:
                forged_images.append(os.path.join(person_folder_path, filename))

        additional_images = random.sample(image_paths, 10)
        forged_images.extend(additional_images)
        num_genuine_images = len(genuine_images)
        num_forged_images = len(forged_images)
        num_combinations = min(num_genuine_images * (num_genuine_images - 1) // 2, num_genuine_images * num_forged_images)
        genuine_combinations = random.sample(list(itertools.combinations(genuine_images, 2)), num_combinations)
        forged_combinations = random.sample(list(itertools.product(genuine_images, forged_images)), num_combinations)

        # Create a DataFrame with the balanced combinations
        data = []
        for (image_1, image_2), (genuine_image, forged_image) in zip(genuine_combinations, forged_combinations):
            anchor_path = os.path.join(image_1)
            positive_path = os.path.join(image_2)
            label = 0
            data.append([anchor_path, positive_path, label])

            anchor_path = os.path.join(genuine_image)
            positive_path = os.path.join(forged_image)
            label = 1
            data.append([anchor_path, positive_path, label])

        df = df.append(pd.DataFrame(data, columns=['Image1', 'Image2', 'Label']), ignore_index=True)
    return df