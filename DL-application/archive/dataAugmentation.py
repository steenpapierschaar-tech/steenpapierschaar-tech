from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from config import config

def display_augmented_samples(original_images, augmented_images, n_samples=5):
    """Display original images next to their augmented versions"""
    # Safety check for empty lists or mismatched sizes
    if not original_images or not augmented_images:
        print("[WARNING] No samples to display")
        return
        
    # Use minimum length to avoid index errors
    n_samples = min(len(original_images), len(augmented_images), n_samples)
    if n_samples == 0:
        print("[WARNING] No samples to display")
        return
        
    plt.figure(figsize=(2*n_samples, 4))
    
    for idx in range(n_samples):
        # Display original image
        plt.subplot(2, n_samples, idx + 1)
        plt.imshow(original_images[idx])
        plt.axis('off')
        if idx == 0:
            plt.title('Original')
            
        # Display augmented image
        plt.subplot(2, n_samples, n_samples + idx + 1)
        plt.imshow(augmented_images[idx])
        plt.axis('off')
        if idx == 0:
            plt.title('Augmented')
    
    plt.tight_layout()
    plt.show()

def augmentData(images, labels, target_size=config.TARGET_AUGMENTATION_SIZE):
    """
    Augment the training data to reach target_size, ensuring balanced classes
    """
    
    # TODO: Add vertical flip option
    # TODO: Add brightness/contrast augmentation
    
    # Calculate number of images to generate
    num_to_generate = target_size - len(images)
    
    # Group images by their labels
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_idx = np.argmax(label)  # Convert one-hot back to index
        label_indices[label_idx].append(idx)
    
    num_classes = len(label_indices)
    images_per_class = num_to_generate // num_classes
    
    print(f"\n[INFO] Augmentation plan:")
    print(f"    Total new images needed: {num_to_generate}")
    print(f"    Number of classes: {num_classes}")
    print(f"    Images to generate per class: {images_per_class}")
    
    # Initialize the ImageDataGenerator with moderate augmentation
    datagen = ImageDataGenerator(
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.SHIFT_RANGE,
        height_shift_range=config.SHIFT_RANGE,
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        fill_mode=config.FILL_MODE
    )
    
    # Generate new images
    augmented_images = []
    augmented_labels = []
    
    # Keep track of some original-augmented pairs for visualization
    vis_originals = []
    vis_augmented = []
    samples_to_show = 5
    
    # Generate images for each class
    for label_idx, indices in label_indices.items():
        print(f"[INFO] Generating {images_per_class} new images for class {label_idx}")
        
        for i in range(images_per_class):
            # Cycle through existing images of this class
            idx = indices[i % len(indices)]
            img = images[idx]
            label = labels[idx]
            
            # Generate a new augmented image
            aug_img = next(datagen.flow(np.expand_dims(img, 0), batch_size=1))[0]
            
            # Only store visualization samples from the first few iterations
            if i < samples_to_show:
                vis_originals.append(img)
                vis_augmented.append(aug_img)
            
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    # Only display samples if we have them
    if vis_originals and vis_augmented:
        display_augmented_samples(vis_originals, vis_augmented)
    
    # Combine original and augmented data
    final_images = np.concatenate([images, np.array(augmented_images)])
    final_labels = np.concatenate([labels, np.array(augmented_labels)])
    
    print(f"\n[INFO] Augmentation complete:")
    print(f"    Original dataset size: {len(images)}")
    print(f"    New images generated: {len(augmented_images)}")
    print(f"    Final dataset size: {len(final_images)}")
    
    return final_images, final_labels
