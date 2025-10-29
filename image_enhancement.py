# image_enhancement_with_metrics.py
"""
Usage:
    python image_enhancement_with_metrics.py --input test_image.jpg --outdir results
    python image_enhancement_with_metrics.py  # uses/creates test_image.jpg
"""

import os
import argparse
import csv
from typing import Tuple, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_img(path: str, img: np.ndarray):
    if img.dtype != np.uint8:
        img_to_save = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_to_save = img
    cv2.imwrite(path, img_to_save)

def valid_odd_kernel(k):
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k

def compute_psnr_ssim(ref: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    if ref.shape != target.shape:
        target = cv2.resize(target, (ref.shape[1], ref.shape[0]))
    if ref.dtype != np.uint8:
        ref = np.clip(ref, 0, 255).astype(np.uint8)
    if target.dtype != np.uint8:
        target = np.clip(target, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(ref, target, data_range=255)
    ssim = structural_similarity(ref, target, data_range=255)
    return psnr, ssim

def compute_image_stats(img: np.ndarray) -> Dict[str, float]:
    return {
        'mean': float(np.mean(img)),
        'std': float(np.std(img)),
        'min': int(np.min(img)),
        'max': int(np.max(img)),
        'dynamic_range': int(np.max(img) - np.min(img))
    }


class ImageEnhancer:
    
    def __init__(self, image_path: str, outdir: str = "results"):
        self.image_path = image_path
        self.outdir = outdir
        ensure_outdir(self.outdir)

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.color = self.image.copy()
        if len(self.color.shape) == 3:
            self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.color.copy()
            self.color = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        self.results = {}
        self.stats = {'original': compute_image_stats(self.gray)}
        
        save_img(os.path.join(self.outdir, "original_input.png"), self.color)
        save_img(os.path.join(self.outdir, "original_gray.png"), self.gray)

    def contrast_stretching(self, min_percentile: float = 2, max_percentile: float = 98):
        min_val = np.percentile(self.gray, min_percentile)
        max_val = np.percentile(self.gray, max_percentile)

        if max_val <= min_val:
            stretched = self.gray.copy()
        else:
            stretched = ((self.gray.astype(np.float32) - min_val) * (255.0 / (max_val - min_val)))
            stretched = np.clip(stretched, 0, 255).astype(np.uint8)

        self.results['contrast_stretching'] = stretched
        self.stats['contrast_stretching'] = compute_image_stats(stretched)
        save_img(os.path.join(self.outdir, 'contrast_stretching.png'), stretched)
        return stretched

    def histogram_equalization(self):
        equalized = cv2.equalizeHist(self.gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(self.gray)

        self.results['histogram_equalization'] = equalized
        self.results['clahe'] = clahe
        self.stats['histogram_equalization'] = compute_image_stats(equalized)
        self.stats['clahe'] = compute_image_stats(clahe)

        save_img(os.path.join(self.outdir, 'histogram_equalization.png'), equalized)
        save_img(os.path.join(self.outdir, 'clahe.png'), clahe)
        return equalized, clahe

    def add_noise(self, noise_type: str = 'gaussian', mean: float = 0.0, std: float = 25.0):
        noisy = self.gray.astype(np.float32)
        if noise_type == 'gaussian':
            noise = np.random.normal(mean, std, self.gray.shape)
            noisy = noisy + noise
        elif noise_type == 'salt_pepper':
            rnd = np.random.random(self.gray.shape)
            noisy[rnd < 0.05] = 0
            noisy[rnd > 0.95] = 255
        else:
            raise ValueError("noise_type must be 'gaussian' or 'salt_pepper'")

        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        self.results['noisy_image'] = noisy
        self.stats['noisy_image'] = compute_image_stats(noisy)
        save_img(os.path.join(self.outdir, 'noisy_image.png'), noisy)
        return noisy

    def remove_noise_gaussian(self, noisy_image: np.ndarray, kernel_size: int = 5):
        k = valid_odd_kernel(kernel_size)
        denoised = cv2.GaussianBlur(noisy_image, (k, k), 0)
        self.results['gaussian_denoised'] = denoised
        self.stats['gaussian_denoised'] = compute_image_stats(denoised)
        save_img(os.path.join(self.outdir, f'gaussian_denoised_k{k}.png'), denoised)
        return denoised

    def remove_noise_median(self, noisy_image: np.ndarray, kernel_size: int = 5):
        k = valid_odd_kernel(kernel_size)
        denoised = cv2.medianBlur(noisy_image, k)
        self.results['median_denoised'] = denoised
        self.stats['median_denoised'] = compute_image_stats(denoised)
        save_img(os.path.join(self.outdir, f'median_denoised_k{k}.png'), denoised)
        return denoised

    def edge_detection_sobel(self):
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_combined = np.clip((sobel_combined / sobel_combined.max()) * 255 if sobel_combined.max() != 0 else sobel_combined, 0, 255).astype(np.uint8)
        self.results['sobel_edges'] = sobel_combined
        save_img(os.path.join(self.outdir, 'sobel_edges.png'), sobel_combined)
        return sobel_combined

    def edge_detection_canny(self, low_threshold: int = 100, high_threshold: int = 200):
        edges = cv2.Canny(self.gray, low_threshold, high_threshold)
        self.results['canny_edges'] = edges
        save_img(os.path.join(self.outdir, 'canny_edges.png'), edges)
        return edges

    def plot_results(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.outdir, 'results_comparison.png')

        display_items = [('Original (RGB)', cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)),
                         ('Original (Gray)', self.gray)]
        for key, img in self.results.items():
            if len(img.shape) == 2:
                display_items.append((key.replace('_', ' ').title(), img))
            else:
                display_items.append((key.replace('_', ' ').title(), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

        n = len(display_items)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten()
        for i, (title, img) in enumerate(display_items):
            ax = axes[i]
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Comparison plot -> {save_path}")
        plt.close(fig)

    def plot_histograms(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.outdir, 'histograms_comparison.png')

        imgs = [('Original', self.gray)]
        if 'contrast_stretching' in self.results:
            imgs.append(('Contrast Stretching', self.results['contrast_stretching']))
        if 'histogram_equalization' in self.results:
            imgs.append(('Histogram Equalization', self.results['histogram_equalization']))
        if 'clahe' in self.results:
            imgs.append(('CLAHE', self.results['clahe']))

        cols = len(imgs)
        fig, axes = plt.subplots(2, cols, figsize=(5*cols, 6))
        for i, (title, img) in enumerate(imgs):
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(title)
            axes[0, i].axis('off')

            axes[1, i].hist(img.flatten(), bins=256, range=[0, 256], color='C0', alpha=0.8)
            axes[1, i].set_title(f"{title} Histogram")
            axes[1, i].set_xlabel('Pixel intensity')
            axes[1, i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Histogram plot -> {save_path}")
        plt.close(fig)

    def compute_and_save_metrics(self, csv_path: str = None):
        """
        Compute PSNR/SSIM metrics for meaningful comparisons only.
        Note: Edge detection results are excluded as they represent transformations
        rather than enhancements.
        """
        if csv_path is None:
            csv_path = os.path.join(self.outdir, "metrics.csv")

        ref = self.gray
        rows = []
        
        enhancement_methods = ['contrast_stretching', 'histogram_equalization', 'clahe']
        for key in enhancement_methods:
            if key in self.results:
                try:
                    psnr, ssim = compute_psnr_ssim(ref, self.results[key])
                    rows.append((key, psnr, ssim, 'vs_original'))
                except Exception:
                    pass

        if 'noisy_image' in self.results:
            noisy = self.results['noisy_image']
            psnr, ssim = compute_psnr_ssim(ref, noisy)
            rows.append(('noisy_image', psnr, ssim, 'vs_original'))

            for den_key in ['gaussian_denoised', 'median_denoised']:
                if den_key in self.results:
                    psnr, ssim = compute_psnr_ssim(ref, self.results[den_key])
                    rows.append((den_key, psnr, ssim, 'restoration_quality'))

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'psnr_db', 'ssim', 'comparison_type'])
            for r in rows:
                writer.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", r[3]])
        print(f"[Saved] Metrics -> {csv_path}")
        
        stats_path = csv_path.replace('metrics.csv', 'statistics.csv')
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'mean', 'std', 'min', 'max', 'dynamic_range'])
            for method, stats in self.stats.items():
                writer.writerow([method, f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
                               stats['min'], stats['max'], stats['dynamic_range']])
        print(f"[Saved] Statistics -> {stats_path}")
        
        return rows


def create_test_image(path: str = 'test_image.jpg', size: int = 500):
    img = np.zeros((size, size), dtype=np.uint8)

    step = size // 5
    for i in range(0, size, step):
        intensity = 60 + (i // step) * 24
        img[i:i+step, :] = intensity

    yy, xx = np.ogrid[:size, :size]
    center_region = slice(size//4, 3*size//4)
    grad = 60 + ((xx + yy) / (2*size) * 120).astype(np.uint8)
    img[center_region, center_region] = grad[center_region, center_region]
    
    cv2.imwrite(path, img)
    print(f"Created test image with limited contrast (range 60-180) at {path}")
    return path

def main():
    p = argparse.ArgumentParser(description="Image Enhancement with Metrics - Academic Version")
    p.add_argument('--input', '-i', default='test_image.jpg', help='Input image path')
    p.add_argument('--outdir', '-o', default='results', help='Output directory')
    p.add_argument('--noisy-std', type=float, default=20.0, help='Gaussian noise std dev')
    p.add_argument('--skip-plots', action='store_true', help='Skip matplotlib plots')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Input '{args.input}' not found — creating synthetic test image.")
        create_test_image(args.input, size=500)

    enhancer = ImageEnhancer(args.input, outdir=args.outdir)

    print("\n=== Applying Enhancement Techniques ===")
    print("1. Contrast Stretching (percentile-based)...")
    enhancer.contrast_stretching(min_percentile=2, max_percentile=98)

    print("2. Histogram Equalization (global + CLAHE)...")
    enhancer.histogram_equalization()

    print(f"3. Adding Gaussian noise (std={args.noisy_std})...")
    noisy = enhancer.add_noise('gaussian', std=args.noisy_std)

    print("4. Gaussian denoising...")
    enhancer.remove_noise_gaussian(noisy, kernel_size=5)

    print("5. Median denoising...")
    enhancer.remove_noise_median(noisy, kernel_size=5)

    print("6. Sobel edge detection...")
    enhancer.edge_detection_sobel()

    print("7. Canny edge detection...")
    enhancer.edge_detection_canny()

    if not args.skip_plots:
        print("\n=== Generating Visualizations ===")
        enhancer.plot_results()
        enhancer.plot_histograms()

    print("\n=== Computing Metrics ===")
    rows = enhancer.compute_and_save_metrics()

    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    print(f"{'Method':<30} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'Type'}")
    print("-"*70)
    for r in rows:
        print(f"{r[0]:<30} | {r[1]:>10.3f} | {r[2]:>10.4f} | {r[3]}")
    
    print("\n" + "="*70)
    print(f"All outputs saved in: {args.outdir}/")
    print("="*70)

if __name__ == "__main__":
    main()