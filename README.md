# Image Reconstruction via Iterative Optimization

This project demonstrates image reconstruction starting from pure random noise.
Instead of traditional compression and decompression, we use iterative optimization to gradually modify a noisy image to match a given target image.

The reconstruction process is monitored over multiple steps using the Root Mean Square Error (RMSE) between the generated image and the target image.


## Highlights

- Start from pure random noise.
- Gradually optimize pixel values to match the target image.
- Track the reconstruction quality at each step using RMSE.
- No explicit compression or decompression involved.
- Visualize the improvement of the image over optimization steps.

## Methodology 

- **Initialization**
    - A random image (noise) is generated.

- **Optimization Loop**
    - In each iteration (or step):
        - Calculate the difference between the generated image and the target image.
        - Update the pixels of the generated image to reduce this difference.
        - Compute and record the RMSE.

- **Stopping Condition**
    - The optimization runs for a fixed number of steps (or until convergence).

- **Visualization**
    - The reconstruction progress is plotted after each step.

## Output Example

- **First Image**: The original target image (downscaled or reduced for this experiment).
- **Second Image (Grid Layout)**: 
    - Shows the gradual improvement of the noisy image.
    - Each small image corresponds to a reconstruction step. 
    The RMSE value shown above each step indicates how close the generated image is to the target.

| Step | Description                                 |
|:----:|:-------------------------------------------:|
|  0   | Random noise (high RMSE)                    |
| 1-3  | Early rough matching of basic patterns      |
| 4-8  | Fine-tuning to closely match the target image |



## Repository Structure

```text
src/
  ├── main.py         # Main script: Runs compression and decompression
  ├── monkey.gif      # Example input image

results/
  ├── Figure_1.png    # The image 
  ├── Figure_2.png    # The reconstruction process 

README.md
```

## References

- [An Introduction to Fractal Image Compression](https://www.ti.com/lit/an/bpra065/bpra065.pdf?ts=1745827705428&ref_url=https%253A%252F%252Fwww.google.com%252F), Literature Number: BPRA065, Texas Instruments Europe, October 1997.

- [Fractal Image Compression](https://pvigier.github.io/2018/05/14/fractal-image-compression.html), pvigier's blog, May 2018.


## License

This project is open-source and available under the [MIT License](LICENSE).
