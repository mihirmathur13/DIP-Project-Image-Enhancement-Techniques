# Image Enhancement Techniques

This project implements common image enhancement techniques including contrast stretching and histogram equalization. I tested these on a foggy morning photograph to show how they can improve low-contrast images.

## What's Inside

- `image_enhancement.py` - Main script with all the enhancement methods
- `foggy.webp` - Test image (foggy morning photo)
- `results/` - Output folder with all processed images and metrics
- `requirements.txt` - Python packages needed to run this

## Setup

Make sure you have Python 3.8 or later installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Code

Basic usage with the included test image:

```bash
python image_enhancement.py --input foggy.webp --outdir results
```

If you want to try it on your own image:

```bash
python image_enhancement.py --input your_image.jpg --outdir output_folder
```

The script will automatically:
- Apply contrast stretching
- Apply histogram equalization (both global and CLAHE)
- Add some noise and then remove it using Gaussian and Median filters
- Detect edges using Sobel and Canny methods
- Generate comparison plots and histograms
- Save metrics to CSV files

## What Gets Generated

After running, check the output folder for:

**Images:**
- `original_gray.png` - Grayscale version of input
- `contrast_stretching.png` - After contrast enhancement
- `histogram_equalization.png` - After global histogram equalization
- `clahe.png` - After adaptive histogram equalization
- Edge detection results (Sobel and Canny)
- Noise addition and removal examples

**Visualizations:**
- `results_comparison.png` - Grid showing all results side by side
- `histograms_comparison.png` - Histograms showing pixel distribution changes

**Data Files:**
- `metrics.csv` - PSNR and SSIM quality metrics
- `statistics.csv` - Mean, std deviation, and dynamic range for each method


## Notes

- The script saves everything as PNG to avoid compression artifacts
- Edge detection images are separate - they're transformations rather than enhancements