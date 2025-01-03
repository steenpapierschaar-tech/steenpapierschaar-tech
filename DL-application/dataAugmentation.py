from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from fileHandler import loadFiles, createAugmentedDirs

def augmentData(filelist, target_size):   
    # Calculate the size of the dataset
    size = len(filelist)
    
    # Calculate the number of images to generate
    new_size = target_size - size
    
    # Print the number of images to be generated
    print(f"Number of images to be generated: {new_size}")
    
    # Ensure the augmented directories exist
    baseDir = os.path.join(os.getcwd(), "photoDataset_aug")
    augDirs = createAugmentedDirs(baseDir)
    
    print("[INFO] Augmented directories created: ", augDirs)
    
    # Initialize the ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Group images by category
    categories = {}
    for img_path in filelist:
        category = os.path.basename(os.path.dirname(img_path))
        if category not in categories:
            categories[category] = []
        categories[category].append(img_path)
    
    # Print found categories and their counts
    for category, img_paths in categories.items():
        print(f"[INFO] Found category '{category}' with {len(img_paths)} images.")
    
    # Calculate the number of images to generate per category
    num_categories = len(categories)
    new_size_per_category = new_size // num_categories
    
    print(f"[INFO] Generating {new_size_per_category} images per category.")
    
    # Generate new images for each category
    for category, img_paths in categories.items():
        augDir = augDirs.get(category, baseDir)
        for i in range(new_size_per_category):
            img_path = img_paths[i % len(img_paths)]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = np.expand_dims(img, 0)
            
            # Get original filename without extension
            orig_filename = os.path.splitext(os.path.basename(img_path))[0]
            
            for batch in datagen.flow(img, batch_size=1, save_to_dir=augDir, 
                                      save_prefix=orig_filename, save_format='png'):
                break
    
    # Update filelist with augmented images
    augmented_files = loadFiles(baseDir)
    filelist.extend(augmented_files)
    
    return filelist