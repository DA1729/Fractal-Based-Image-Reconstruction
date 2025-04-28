import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import optimize
import math

# Image channel operations

def convert_to_grayscale(image_data):
    return np.mean(image_data[:,:,:2], axis=2)

def separate_color_channels(image_data):
    return image_data[:,:,0], image_data[:,:,1], image_data[:,:,2]

def combine_color_channels(red_channel, green_channel, blue_channel):
    shape = (red_channel.shape[0], red_channel.shape[1], 1)
    return np.concatenate((
        np.reshape(red_channel, shape), 
        np.reshape(green_channel, shape),
        np.reshape(blue_channel, shape)), axis=2)

# Image transformations

def downsample_image(image_data, scale_factor):
    new_shape = (image_data.shape[0] // scale_factor, image_data.shape[1] // scale_factor)
    result = np.zeros(new_shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            patch = image_data[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor]
            result[i,j] = np.mean(patch)
    return result

def apply_rotation(image_data, rotation_angle):
    return ndimage.rotate(image_data, rotation_angle, reshape=False)

def apply_flip(image_data, flip_direction):
    return image_data[::flip_direction,:]

def transform_image(image_data, flip_direction, rotation_angle, contrast=1.0, brightness=0.0):
    flipped = apply_flip(image_data, flip_direction)
    rotated = apply_rotation(flipped, rotation_angle)
    return contrast * rotated + brightness

# Image adjustment parameters

def calculate_adjustment_parameters_simple(target, source):
    contrast = 0.75
    brightness = (np.sum(target - contrast*source)) / target.size
    return contrast, brightness 

def calculate_adjustment_parameters(target, source):
    X = np.concatenate((np.ones((source.size, 1)), 
                       np.reshape(source, (source.size, 1))), axis=1)
    y = np.reshape(target, (target.size,))
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return params[1], params[0]

# Grayscale compression

def generate_transformation_candidates(image_data, src_size, dst_size, step_size):
    scaling = src_size // dst_size
    candidates = []
    
    for i in range((image_data.shape[0] - src_size) // step_size + 1):
        for j in range((image_data.shape[1] - src_size) // step_size + 1):
            source_block = downsample_image(
                image_data[i*step_size:i*step_size+src_size, j*step_size:j*step_size+src_size], 
                scaling)
                
            for flip_dir, rot_angle in transformation_options:
                transformed = transform_image(source_block, flip_dir, rot_angle)
                candidates.append((i, j, flip_dir, rot_angle, transformed))
    return candidates

def perform_compression(image_data, src_size, dst_size, step_size):
    transformation_rules = []
    candidate_transforms = generate_transformation_candidates(image_data, src_size, dst_size, step_size)
    
    rows = image_data.shape[0] // dst_size
    cols = image_data.shape[1] // dst_size
    
    for row in range(rows):
        transformation_rules.append([])
        for col in range(cols):
            print(f"Processing block {row+1}/{rows}, {col+1}/{cols}")
            transformation_rules[row].append(None)
            
            target_block = image_data[
                row*dst_size:(row+1)*dst_size,
                col*dst_size:(col+1)*dst_size]
            
            min_error = float('inf')
            
            for i, j, flip_dir, rot_angle, source in candidate_transforms:
                contrast, brightness = calculate_adjustment_parameters(target_block, source)
                adjusted_source = contrast * source + brightness
                error = np.sum(np.square(target_block - adjusted_source))
                
                if error < min_error:
                    min_error = error
                    transformation_rules[row][col] = (
                        i, j, flip_dir, rot_angle, contrast, brightness)
    return transformation_rules

def reconstruct_image(transformation_rules, src_size, dst_size, step_size, iterations=8):
    scaling = src_size // dst_size
    height = len(transformation_rules) * dst_size
    width = len(transformation_rules[0]) * dst_size
    
    reconstruction_steps = [np.random.randint(0, 256, (height, width))]
    current_image = np.zeros((height, width))
    
    for iteration in range(iterations):
        print(f"Reconstruction iteration {iteration+1}/{iterations}")
        
        for i in range(len(transformation_rules)):
            for j in range(len(transformation_rules[i])):
                rule = transformation_rules[i][j]
                src_i, src_j, flip_dir, rot_angle, contrast, brightness = rule
                
                source_region = reconstruction_steps[-1][
                    src_i*step_size:src_i*step_size+src_size,
                    src_j*step_size:src_j*step_size+src_size]
                
                downsampled = downsample_image(source_region, scaling)
                transformed = transform_image(downsampled, flip_dir, rot_angle, contrast, brightness)
                
                current_image[
                    i*dst_size:(i+1)*dst_size,
                    j*dst_size:(j+1)*dst_size] = transformed
        
        reconstruction_steps.append(current_image)
        current_image = np.zeros((height, width))
    
    return reconstruction_steps

# Color image compression

def downsample_color_image(image_data, factor):
    r, g, b = separate_color_channels(image_data)
    r = downsample_image(r, factor)
    g = downsample_image(g, factor)
    b = downsample_image(b, factor)
    return combine_color_channels(r, g, b)

def compress_color_image(image_data, src_size, dst_size, step_size):
    r, g, b = separate_color_channels(image_data)
    return [
        perform_compression(r, src_size, dst_size, step_size),
        perform_compression(g, src_size, dst_size, step_size),
        perform_compression(b, src_size, dst_size, step_size)
    ]

def reconstruct_color_image(transformations, src_size, dst_size, step_size, iterations=8):
    r = reconstruct_image(transformations[0], src_size, dst_size, step_size, iterations)[-1]
    g = reconstruct_image(transformations[1], src_size, dst_size, step_size, iterations)[-1]
    b = reconstruct_image(transformations[2], src_size, dst_size, step_size, iterations)[-1]
    return combine_color_channels(r, g, b)

# Visualization

def display_reconstruction_steps(reconstruction_steps, reference_image=None):
    plt.figure(figsize=(12, 8))
    steps = len(reconstruction_steps)
    grid_size = math.ceil(math.sqrt(steps))
    
    for idx, image in enumerate(reconstruction_steps):
        plt.subplot(grid_size, grid_size, idx+1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='none')
        
        if reference_image is None:
            plt.title(f'Step {idx}')
        else:
            error = np.sqrt(np.mean(np.square(reference_image - image)))
            plt.title(f'Step {idx} (RMSE: {error:.2f})')
        
        plt.axis('off')
    
    plt.tight_layout()

# Configuration

transformation_options = [
    [direction, angle] 
    for direction in [1, -1] 
    for angle in [0, 90, 180, 270]
]

# Example usage

def demonstrate_grayscale_compression():
    original_image = plt.imread('monkey.gif')
    grayscale_image = convert_to_grayscale(original_image)
    reduced_image = downsample_image(grayscale_image, 4)
    
    plt.figure()
    plt.imshow(reduced_image, cmap='gray', interpolation='none')
    plt.title('Original (Reduced)')
    
    compression_rules = perform_compression(reduced_image, 8, 4, 8)
    reconstruction = reconstruct_image(compression_rules, 8, 4, 8)
    
    display_reconstruction_steps(reconstruction, reduced_image)
    plt.show()

def demonstrate_color_compression():
    color_image = plt.imread('lena.gif')
    reduced_color = downsample_color_image(color_image, 8)
    
    compression_rules = compress_color_image(reduced_color, 8, 4, 8)
    reconstructed_color = reconstruct_color_image(compression_rules, 8, 4, 8)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(reduced_color).astype(np.uint8), interpolation='none')
    plt.title('Original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_color.astype(np.uint8), interpolation='none')
    plt.title('Reconstructed')
    plt.show()

if __name__ == '__main__':
    demonstrate_grayscale_compression()
    # demonstrate_color_compression()