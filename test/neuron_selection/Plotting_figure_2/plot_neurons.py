import pandas as pd
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate

# --- OPTIMIZED IMPORTS ---
from scipy.signal import fftconvolve # Use FFT for speed
from skimage.transform import resize # Use for high-quality downsampling

# --- PLOT STYLING ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16 # Adjusted for better subplot display
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16


# --- Unchanged Analysis Functions ---
def calculate_spatial_information(ratemap, occupancy_map):
    """Calculates the spatial information (bits per spike) in a numerically stable way."""
    mean_rate = np.sum(ratemap * occupancy_map)
    if mean_rate == 0: return 0.0
    safe_bins = (occupancy_map > 0) & (ratemap > 0)
    if not np.any(safe_bins): return 0.0
    p_i = occupancy_map[safe_bins] / np.sum(occupancy_map[safe_bins])
    lambda_i = ratemap[safe_bins]
    return np.sum(p_i * (lambda_i / mean_rate) * np.log2(lambda_i / mean_rate))

def calculate_borderness_score(ratemap, border_width=0.02): # Reduced width to 0.1 for precision
    """
    Calculates the borderness score by checking each wall individually 
    and returning the maximum score found.
    """
    # 1. Handle Negative Values: Shift data so minimum is 0 to prevent math errors
    if np.min(ratemap) < 0:
        ratemap = ratemap - np.min(ratemap)
        
    h, w = ratemap.shape
    b_pix_h = int(h * border_width)
    b_pix_w = int(w * border_width)
    
    # 2. Define the CENTER mask (same for all)
    center_mask = np.ones_like(ratemap, dtype=bool)
    center_mask[:b_pix_h, :] = False
    center_mask[-b_pix_h:, :] = False
    center_mask[:, :b_pix_w] = False
    center_mask[:, -b_pix_w:] = False
    
    mean_center = np.mean(ratemap[center_mask]) if np.any(center_mask) else 0

    # 3. Calculate score for EACH wall separately
    walls = {
        "Top": ratemap[-b_pix_h:, :],    # Python arrays: 0 is bottom, -1 is top usually (depends on origin)
        "Bottom": ratemap[:b_pix_h, :],
        "Left": ratemap[:, :b_pix_w],
        "Right": ratemap[:, -b_pix_w:]
    }
    
    scores = []
    
    for wall_name, wall_data in walls.items():
        mean_border = np.mean(wall_data)
        
        # Avoid division by zero
        if (mean_border + mean_center) == 0:
            score = 0.0
        else:
            score = (mean_border - mean_center) / (mean_border + mean_center)
        scores.append(score)
        
    # 4. Return the Maximum score among the 4 walls
    return max(scores)

def calculate_gridness_score_fast(ratemap, downsample_size=128):
    """Calculates the gridness score using downsampling and FFT-based autocorrelation."""
    ratemap_small = resize(ratemap, (downsample_size, downsample_size), anti_aliasing=True)
    ratemap_no_nan = np.nan_to_num(ratemap_small)
    autocorr = fftconvolve(ratemap_no_nan, ratemap_no_nan[::-1, ::-1], mode='same')
    h, w = autocorr.shape
    center_y, center_x = h // 2, w // 2
    central_peak_mask = autocorr > (autocorr[center_y, center_x] * 0.5)
    autocorr[central_peak_mask] = 0
    corrs = []
    angles = [30, 60, 90, 120, 150]
    for angle in angles:
        rotated_autocorr = rotate(autocorr, angle, reshape=False, order=0)
        valid_mask = ~np.isnan(autocorr) & ~np.isnan(rotated_autocorr)
        corr = np.corrcoef(autocorr[valid_mask], rotated_autocorr[valid_mask])[0, 1]
        corrs.append(corr)
    if len(corrs) < 5: return -1.0
    gridness = min(corrs[1], corrs[3]) - max(corrs[0], corrs[2], corrs[4])
    return gridness

def classify_cell(si, g, b):
    """Infers cell type based on scores."""
    if g > 0.35: return "Grid Cell"
    elif b > 0.5: return "Border Cell"
    elif si > 1.0 and g < 0.2 and b < 0.2: return "Place Cell"
    elif si > 0.5: return "Spatially Modulated Cell"
    else: return "Unclassified"
# --- Main script ---
if __name__ == "__main__":
    # 1. Define the two files to be plotted


    # other cells
    file1 = "./border_20250504/ratemap_data_neuron_446.npy"
    # file2 = "./border_20250504/ratemap_data_neuron_770.npy"



    files_to_plot = [file1]
    
    ratemaps_display = []
    titles = []

    # UNIT CONVERSION SETTINGS
    # pixel_to_meter = 10  # 1 pixel = 10 meters
    pixel_to_km = 10 / 1000  # 1 pixel = 10 meters = 0.01 kilometers
    
    # Load first map to get occupancy and shape info
    temp_map = np.load(files_to_plot[0])
    # temp_map = temp_map[60:1070, 45:805]
    temp_map = temp_map[60:1070, 70:820]
    occupancy_map = np.ones_like(temp_map) / temp_map.size
    
    # Calculate physical dimensions for plotting
    h_pixels, w_pixels = temp_map.shape
    h_km = h_pixels * pixel_to_km
    w_km = w_pixels * pixel_to_km


    # Define extent: [left, right, bottom, top] in data coordinates
    # origin='lower' puts (0,0) at bottom-left, so we go from 0 to max width/height
    # plot_extent = [0, w_meters, 0, h_meters]
    plot_extent = [0, w_km, 0, h_km]

    # 2. Process each file
    for idx, file in enumerate(files_to_plot):
        ratemap = np.load(file)
        ratemap = ratemap[60:1070, 70:820] # Apply the slice
        
        # if idx == 1:
        #     ratemap = ratemap-0.8
        # if idx == 0:
        #     ratemap = ratemap+0.3

        # Smooth
        ratemap_smoothed = gaussian_filter(ratemap, sigma=1)
        ratemaps_display.append(ratemap_smoothed)




        neuron_id = os.path.basename(file).split('_')[-1].split('.')[0]
        title_str = (f"Neuron {neuron_id}\n")
        titles.append(title_str)

    # 3. Calculate shared color scale
    vmin = min(r.min() for r in ratemaps_display)
    vmax = max(r.max() for r in ratemaps_display)
    # vmin = 0
    # vmax = 0.5
    print(f"Using shared color scale from {vmin:.2f} to {vmax:.2f}")

    # 4. Create the plot
    fig, axs = plt.subplots(1, 1, figsize=(18, 8), sharey=True)
    fig.suptitle("Comparison of Neuronal Activity Ratemaps", fontsize=22)

    # Plot neuron 1
    # Note: added 'extent' parameter here
    im = axs.imshow(ratemaps_display[0], cmap='jet', origin='lower', 
                       vmin=vmin, vmax=vmax, extent=plot_extent)

    axs.set_xlabel("X Position (km)")
    axs.set_ylabel("Y Position (km)")

    # # Plot neuron 2
    # # Note: added 'extent' parameter here
    # axs[1].imshow(ratemaps_display[1], cmap='jet', origin='lower', 
    #               vmin=vmin, vmax=vmax, extent=plot_extent)
    # axs[1].set_xlabel("X Position (m)")

    # 5. Add shared colorbar
    fig.colorbar(im, label="Average Activation",  pad=0.02)

    # 6. Adjust layout
    plt.tight_layout(rect=[0, 0.03, 0.68, 0.96]) 
    
    plt.show()
    plt.close()