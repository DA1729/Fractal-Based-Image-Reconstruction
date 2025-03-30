import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# preprocessing the image 

def load_and_preprocess(image_path, size = 128):
    """this function loads in the image and does all the preprocessing, i.e., resizing to 128x128 and conversion to grayscale"""
    img = Image.open(image_path).convert("L")
    img = img.resize((size, size), Image.BOX)
    return np.array(img)


# downsampling of the image

def downsample(image_array, factor = 2):
    """creates the downsampled/domain image"""
    h, w = image_array.shape
    return image_array.reshape(h//factor, factor, w//factor, factor).mean(axis = (1, 3))


def split_blocks(image_array, block_size):
    """split image into non-overlapping blocks."""
    h, w = image_array.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image_array[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                blocks.append((i, j, block))
    return blocks

def compute_transformation(domain_block, range_block):
    """Calculate optimal α and t_o with shape matching."""

    upscaled_D = np.kron(domain_block, np.ones((2, 2)))  # repeats each pixel 2x2
    
    D = upscaled_D.flatten()
    R = range_block.flatten()
    N = len(D)
    
    sum_D = np.sum(D)
    sum_R = np.sum(R)
    sum_D2 = np.sum(D**2)
    sum_DR = np.sum(D * R)


    denominator = N * sum_D2 - sum_D**2
    if denominator == 0:
        alpha = 0
    else:
        alpha = (N * sum_DR - sum_D * sum_R) / denominator


    t_o = (sum_R - alpha * sum_D) / N

    # applying transformation
    transformed_block = alpha * upscaled_D + t_o

    # distortion
    distortion = np.sum((transformed_block - range_block)**2)

    return alpha, t_o, distortion


# fractal encoding
def fractal_encode(range_image, domain_image, block_size=4):
    """main encoding function: match domain blocks to range blocks."""
    range_blocks = split_blocks(range_image, block_size)
    domain_blocks = split_blocks(domain_image, block_size // 2)  # domain is downsampled

    codebook = []
    for ri, rj, r_block in range_blocks:
        min_distortion = float('inf')
        best_params = None

        for di, dj, d_block in domain_blocks:
            alpha, to, distortion = compute_transformation(d_block, r_block)
            if distortion < min_distortion:
                min_distortion = distortion
                best_params = (di, dj, alpha, to)

        codebook.append((ri, rj, *best_params))

    return codebook


def fractal_decode(codebook, initial_image, iterations=7, block_size=4):
    """Reconstruct image from codebook."""
    decoded = initial_image.copy()
    domain_size = initial_image.shape[0] // 2
    
    for _ in range(iterations):
        domain_image = downsample(decoded, 2)
        
        for entry in codebook:
            ri, rj, di, dj, alpha, to = entry
            di, dj = int(di), int(dj)  # converting to integer indices
            
            
            domain_block = domain_image[di:di+2, dj:dj+2]
            upscaled = np.kron(domain_block, np.ones((2, 2)))
            
            # Aapplying transformation
            decoded[ri:ri+block_size, rj:rj+block_size] = alpha * upscaled + to
            
    return np.clip(decoded, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # load and preprocess
    range_image = load_and_preprocess("lena.png", 128)
    domain_image = downsample(range_image, 2)

    # encode
    print("encoding...")
    block_size = 4
    codebook = fractal_encode(range_image, domain_image, block_size)

    # savinthe codebook
    np.save("fractal_codebook.npy", codebook)
    print(f"Encoding complete. Codebook size: {len(codebook)} entries.")

    print("decoding...")
    initial_image = np.random.rand(*range_image.shape) * 255
    decoded_image = fractal_decode(codebook, initial_image)

    Image.fromarray(decoded_image).save("decoded_result.jpg")  # Save decoded image
    Image.fromarray(range_image.astype(np.uint8)).save("original.jpg")  # Save original


    # visualize
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(range_image, cmap='gray'), plt.title("Original")
    plt.subplot(132), plt.imshow(initial_image, cmap='gray'), plt.title("Initial (Noise)")
    plt.subplot(133), plt.imshow(decoded_image, cmap='gray'), plt.title("Decoded")
    plt.show()
