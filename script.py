"""
CSS/OSS Measurement Algorithm 
=================================================

This script measures the Close Side Setting (CSS) and Open Side Setting (OSS)
of a gyratory crusher  using LiDAR point cloud data.

Key Innovation: Absolute Height Wall Detection
    - Wall boundary is detected where surface reaches a height threshold
    - This approach ignores floor noise completely and provides robust detection

Algorithm Overview:
    1. Load CSV point cloud data (X, Y, Z coordinates)
    2. Perform axis alignment (90° Y-rotation + translation normalization)
    3. Build a Digital Elevation Model (DEM) via interpolation
    4. Automatically detect the mantle peak (highest central point)
    5. Detect floor level using histogram analysis
    6. Ray-cast from peak to detect mantle edge (gradient-based)
    7. Ray-cast to detect wall boundary (height threshold-based)
    8. Apply smoothing (circle fitting + Savitzky-Golay filter)
    9. Calculate CSS (minimum gap) and OSS (maximum gap)
    10. Generate interactive 3D visualizations

Dependencies:
    - pandas
    - numpy
    - scipy 
    - plotly
    - os
    - sys

Authors: Behlah Katleriwala, Yassh Singh, Mayank Mehan, Anjali Singh 
"""

import pandas as pd
import numpy as np
import os
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter, maximum_filter, sobel
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import sys

# Configure stdout for UTF-8 encoding (required for emoji/special chars)
sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Wall Height Threshold (meters)
# -----------------------------
# This is the KEY parameter for wall detection.
# The wall boundary is detected where the surface height exceeds this value.
# Adjust based on your crusher geometry:
#   - Lower value = wall detected closer to the mantle
#   - Higher value = wall detected further from the mantle
WALL_HEIGHT_THRESHOLD = 0.35  # 35 cm


# =============================================================================
# AXIS ALIGNMENT FUNCTION
# =============================================================================

def rotate_y_90(points):
    """
    Rotate point cloud 90 degrees around the Y-axis.
    
    This transformation aligns the LiDAR scan coordinate system with the
    crusher's natural orientation where Z represents height.
    
    Args:
        points: numpy array of shape (N, 3) containing XYZ coordinates
        
    Returns:
        Rotated points with the same shape
        
    Transformation Matrix:
        | 0  0  1 |   | x |   |  z |
        | 0  1  0 | × | y | = |  y |
        |-1  0  0 |   | z |   | -x |
    """
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    return points @ R.T


# =============================================================================
# CIRCLE FITTING FOR CONTOUR SMOOTHING
# =============================================================================

def fit_circle_to_radii(radii, angles_deg, peak_x, peak_y):
    """
    Fit a circle to radial measurements for contour smoothing.
    
    This function uses algebraic circle fitting (least squares) to find
    the best-fit circle through the measured radii at each angle.
    This produces smooth, circular contours that represent the idealized
    crusher geometry.
    
    Args:
        radii: Array of radial distances from peak (360 values, may contain NaN)
        angles_deg: Array of angles in degrees (0-359)
        peak_x, peak_y: Center point coordinates (mantle peak)
        
    Returns:
        fitted_radii: Smoothed radii at each angle based on fitted circle
        circle_params: Tuple (cx, cy, r) - center and radius of fitted circle
        
    Method:
        1. Convert radii to Cartesian points (x, y)
        2. Solve linear system for circle parameters:
           (x - cx)² + (y - cy)² = r²
           Expanded: 2*cx*x + 2*cy*y + (r² - cx² - cy²) = x² + y²
        3. Compute fitted radius at each angle from the circle equation
    """
    # Filter out NaN values for fitting
    valid = ~np.isnan(radii)
    if np.sum(valid) < 10:
        # Not enough points - return original with fallback
        return radii, (peak_x, peak_y, np.nanmean(radii))
    
    # Convert polar to Cartesian coordinates
    rads = np.radians(angles_deg[valid])
    x_pts = peak_x + radii[valid] * np.cos(rads)
    y_pts = peak_y + radii[valid] * np.sin(rads)
    
    # Algebraic circle fitting using least squares
    # Equation: 2*cx*x + 2*cy*y + c = x² + y²
    # where c = r² - cx² - cy²
    A = np.column_stack([x_pts, y_pts, np.ones_like(x_pts)])
    b = x_pts**2 + y_pts**2
    
    try:
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        cx = result[0] / 2
        cy = result[1] / 2
        r = np.sqrt(result[2] + cx**2 + cy**2)
    except:
        # Fallback to simple mean if fitting fails
        cx, cy = np.mean(x_pts), np.mean(y_pts)
        r = np.mean(np.sqrt((x_pts - cx)**2 + (y_pts - cy)**2))
    
    # Compute fitted radius at each angle
    all_rads = np.radians(angles_deg)
    fitted_radii = []
    
    for theta in all_rads:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Vector from circle center to peak
        dx = peak_x - cx
        dy = peak_y - cy
        
        # Solve quadratic equation for intersection with circle
        # along ray from peak at angle theta
        a = 1
        b_coef = 2 * (dx * cos_t + dy * sin_t)
        c = dx**2 + dy**2 - r**2
        
        discriminant = b_coef**2 - 4*a*c
        if discriminant >= 0:
            t1 = (-b_coef + np.sqrt(discriminant)) / (2*a)
            t2 = (-b_coef - np.sqrt(discriminant)) / (2*a)
            t = max(t1, t2) if t1 > 0 or t2 > 0 else np.nanmean(radii)
            if t < 0:
                t = np.nanmean(radii)
            fitted_radii.append(t)
        else:
            fitted_radii.append(np.nanmean(radii))
    
    return np.array(fitted_radii), (cx, cy, r)


# =============================================================================
# AUTOMATIC MANTLE PEAK DETECTION
# =============================================================================

def detect_mantle_peak(xi, yi, zi_smooth):
    """
    Automatically detect the mantle peak (top of the cone).
    
    The mantle peak is the highest point on the central rotating cone.
    This function finds it by combining height and centrality scoring.
    
    Args:
        xi, yi: Meshgrid arrays for X and Y coordinates
        zi_smooth: Smoothed height (DEM) values on the grid
        
    Returns:
        Tuple (peak_x, peak_y, peak_z): Coordinates of detected peak
        
    Algorithm:
        1. Find all local maxima in the height map
        2. Score each candidate based on:
           - Height (70% weight): Higher is better
           - Centrality (30% weight): Closer to center is better
        3. Return the candidate with the highest combined score
    """
    # Find local maxima using maximum filter
    local_max = maximum_filter(zi_smooth, size=5)
    peaks_mask = (zi_smooth == local_max) & (zi_smooth > 0)
    peak_indices = np.argwhere(peaks_mask)
    
    if len(peak_indices) == 0:
        # Fallback: use global maximum
        idx = np.unravel_index(np.nanargmax(zi_smooth), zi_smooth.shape)
        return xi[idx], yi[idx], zi_smooth[idx]

    # Calculate center of the scan area
    center_x = (xi.min() + xi.max()) / 2
    center_y = (yi.min() + yi.max()) / 2
    
    # Score each peak candidate
    best_score, best_peak = -1, None
    diagonal = np.sqrt((xi.max()-xi.min())**2 + (yi.max()-yi.min())**2)
    
    for idx in peak_indices:
        px = xi[idx[0], idx[1]]
        py = yi[idx[0], idx[1]]
        pz = zi_smooth[idx[0], idx[1]]
        
        # Calculate distance from center
        dist = np.sqrt((px - center_x)**2 + (py - center_y)**2)
        
        # Normalize scores (0 to 1)
        height_norm = pz / zi_smooth.max()
        dist_norm = 1 - (dist / diagonal)
        
        # Combined score: 70% height, 30% centrality
        score = 0.7 * height_norm + 0.3 * dist_norm
        
        if score > best_score:
            best_score = score
            best_peak = (px, py, pz)
            
    return best_peak


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

# ---------------------------------------------------------
# STEP 1: LOAD AND PREPROCESS DATA
# ---------------------------------------------------------
print("=" * 60)
print(" CSS/OSS MEASUREMENT")
print("=" * 60)

# Load point cloud from CSV file
csv_path = r"Test1.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

# Read CSV and normalize column names
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]
points = df[['x', 'y', 'z']].astype(float).values

# Apply axis alignment transformation
# 1. Rotate 90° around Y-axis to orient Z as height
# 2. Translate to set origin at minimum X and Z
points = rotate_y_90(points)
points[:, 0] -= np.min(points[:, 0])  # Shift X to start at 0
points[:, 2] -= np.min(points[:, 2])  # Shift Z to start at 0

# Extract coordinate arrays
x, y, z = points[:, 0], points[:, 1], points[:, 2]

print(f"\n✓ Loaded {len(points)} points from {csv_path}")
print(f"  X range: {x.min():.3f} to {x.max():.3f} m")
print(f"  Y range: {y.min():.3f} to {y.max():.3f} m")
print(f"  Z range: {z.min():.3f} to {z.max():.3f} m")

# ---------------------------------------------------------
# STEP 2: BUILD DIGITAL ELEVATION MODEL (DEM)
# ---------------------------------------------------------
print("\n--- Building Digital Elevation Model ---")

# DEM parameters
grid_res = 400  # Grid resolution (400x400 cells)
sigma = 2       # Gaussian smoothing kernel size

# Create regular grid over XY extent
xi_arr = np.linspace(x.min(), x.max(), grid_res)
yi_arr = np.linspace(y.min(), y.max(), grid_res)
xi, yi = np.meshgrid(xi_arr, yi_arr)

# Interpolate Z values onto the grid using linear interpolation
zi = griddata((x, y), z, (xi, yi), method="linear", fill_value=0)

# Apply Gaussian smoothing to reduce noise
zi_smooth = gaussian_filter(zi, sigma=sigma)

# Compute gradient magnitude for edge detection
gx = sobel(zi_smooth, axis=1)  # X-direction gradient
gy = sobel(zi_smooth, axis=0)  # Y-direction gradient
grad_mag = np.hypot(gx, gy)    # Gradient magnitude

# Create interpolators for ray casting
# Note: RegularGridInterpolator expects (row, col) = (y, x) order
interp_z = RegularGridInterpolator(
    (yi_arr, xi_arr), zi_smooth,
    bounds_error=False, fill_value=np.nan
)
interp_grad = RegularGridInterpolator(
    (yi_arr, xi_arr), grad_mag,
    bounds_error=False, fill_value=0.0
)

# ---------------------------------------------------------
# STEP 3: DETECT MANTLE PEAK
# ---------------------------------------------------------
peak_x, peak_y, peak_z = detect_mantle_peak(xi, yi, zi_smooth)
print(f"✓ Mantle Peak detected: ({peak_x:.3f}, {peak_y:.3f}, {peak_z:.3f}) m")

# ---------------------------------------------------------
# STEP 4: DETECT FLOOR LEVEL
# ---------------------------------------------------------
# Use histogram to find the most common Z value (floor)
hist, bin_edges = np.histogram(z, bins=100)
floor_idx = np.argmax(hist)
floor_z = (bin_edges[floor_idx] + bin_edges[floor_idx + 1]) / 2

# Set measurement height slightly above floor
z_measure = floor_z + 0.02  # 2cm above floor
plane_tol = 0.015           # Tolerance for height filtering

print(f"✓ Floor detected at Z = {floor_z:.3f} m")
print(f"✓ Measurement plane at Z = {z_measure:.3f} m")
print(f"✓ Wall threshold: {WALL_HEIGHT_THRESHOLD:.3f} m")

# ---------------------------------------------------------
# STEP 5: MANTLE EDGE DETECTION (Ray Casting)
# ---------------------------------------------------------
print("\n--- Mantle Edge Detection ---")

# Cast 360 rays outward from the peak (1 per degree)
angles = np.arange(0, 360, 1)
num_samples = 200           # Points per ray
max_search_radius = 0.20    # Maximum search distance (20cm)

mantle_radii = []

for angle_deg in angles:
    theta = np.radians(angle_deg)
    
    # Generate points along ray from peak
    r_samples = np.linspace(0.01, max_search_radius, num_samples)
    x_ray = peak_x + np.cos(theta) * r_samples
    y_ray = peak_y + np.sin(theta) * r_samples
    
    # Sample height and gradient along ray
    z_ray = interp_z(np.c_[y_ray, x_ray])
    grad_ray = interp_grad(np.c_[y_ray, x_ray])
    
    # Find points within mantle height range
    mantle_mask = (z_ray >= floor_z + 0.005) & (z_ray <= z_measure + plane_tol * 2)
    
    if np.any(mantle_mask):
        valid_indices = np.where(mantle_mask)[0]
        
        if len(valid_indices) > 5:
            # Use gradient to find edge (high gradient = edge)
            grad_in_region = grad_ray[valid_indices]
            grad_threshold = np.percentile(grad_in_region, 70)
            high_grad = grad_in_region > grad_threshold
            
            if np.any(high_grad):
                # First high-gradient point is the edge
                edge_local_idx = np.where(high_grad)[0][0]
                edge_idx = valid_indices[edge_local_idx]
            else:
                edge_idx = valid_indices[-1]
        else:
            edge_idx = valid_indices[-1] if len(valid_indices) > 0 else 0
        
        mantle_radii.append(r_samples[edge_idx])
    else:
        mantle_radii.append(np.nan)

mantle_radii = np.array(mantle_radii)
print(f"✓ Mantle radius range: {np.nanmin(mantle_radii)*100:.1f} - {np.nanmax(mantle_radii)*100:.1f} cm")

# ---------------------------------------------------------
# STEP 6: WALL DETECTION (Absolute Height Threshold)
# ---------------------------------------------------------
print("\n--- Wall Detection (Height Threshold Method) ---")

wall_radii = []

for angle_deg in angles:
    theta = np.radians(angle_deg)
    
    # Start search from just past the mantle edge
    mantle_r = mantle_radii[angle_deg] if not np.isnan(mantle_radii[angle_deg]) else 0.05
    
    # Cast ray from mantle edge outward
    r_samples = np.linspace(mantle_r + 0.01, 0.35, 200)
    x_ray = peak_x + np.cos(theta) * r_samples
    y_ray = peak_y + np.sin(theta) * r_samples
    
    # Sample height along ray
    z_ray = interp_z(np.c_[y_ray, x_ray])
    
    # KEY INNOVATION: Wall is where height exceeds threshold
    # This ignores floor noise completely!
    above_wall_height = z_ray > WALL_HEIGHT_THRESHOLD
    
    if np.any(above_wall_height):
        # First point that reaches wall height = wall boundary
        first_wall_idx = np.where(above_wall_height)[0][0]
        wall_r = r_samples[first_wall_idx]
        wall_radii.append(wall_r)
    else:
        wall_radii.append(np.nan)

wall_radii = np.array(wall_radii)
valid_wall = ~np.isnan(wall_radii)
print(f"✓ Valid wall points: {np.sum(valid_wall)} / 360")
if np.sum(valid_wall) > 0:
    print(f"✓ Wall radius range: {np.nanmin(wall_radii)*100:.1f} - {np.nanmax(wall_radii)*100:.1f} cm")
else:
    print("⚠ WARNING: No wall points detected! Try lowering WALL_HEIGHT_THRESHOLD")

# ---------------------------------------------------------
# STEP 7: CONTOUR SMOOTHING
# ---------------------------------------------------------
print("\n--- Contour Smoothing ---")

# Stage 1: Circle fitting
mantle_fitted, mantle_circle = fit_circle_to_radii(mantle_radii, angles, peak_x, peak_y)
wall_fitted, wall_circle = fit_circle_to_radii(wall_radii, angles, peak_x, peak_y)

print(f"✓ Mantle fitted circle radius: {mantle_circle[2]*100:.1f} cm")
print(f"✓ Wall fitted circle radius: {wall_circle[2]*100:.1f} cm")

# Stage 2: Interpolate NaN values and apply Savitzky-Golay filter
mantle_interp = pd.Series(mantle_radii).interpolate(limit_direction='both')
mantle_interp = mantle_interp.fillna(np.nanmedian(mantle_radii))
wall_interp = pd.Series(wall_radii).interpolate(limit_direction='both')
wall_interp = wall_interp.fillna(np.nanmedian(wall_radii))

# Savitzky-Golay filter for additional smoothing
mantle_savgol = savgol_filter(mantle_interp, 51, 3)  # window=51, polynomial=3
wall_savgol = savgol_filter(wall_interp, 51, 3)

# Stage 3: Blend circle fit (70%) with Savgol (30%)
blend = 0.7
mantle_smooth = blend * mantle_fitted + (1 - blend) * mantle_savgol
wall_smooth = blend * wall_fitted + (1 - blend) * wall_savgol

# ---------------------------------------------------------
# STEP 8: CSS/OSS CALCULATION
# ---------------------------------------------------------
print("\n" + "=" * 55)
print("         CSS / OSS RESULTS")
print("=" * 55)

# Calculate gap at each angle
gaps = wall_smooth - mantle_smooth

# Handle any invalid (negative) gaps
valid_gaps = gaps > 0
if not np.all(valid_gaps):
    print(f"⚠ Warning: {np.sum(~valid_gaps)} invalid gaps detected")
    gaps_clean = np.where(valid_gaps, gaps, np.nan)
else:
    gaps_clean = gaps

# Find CSS (minimum gap) and OSS (maximum gap)
css_idx = np.nanargmin(gaps_clean)  # Angle of minimum gap
oss_idx = np.nanargmax(gaps_clean)  # Angle of maximum gap

css_cm = gaps_clean[css_idx] * 100  # Convert to centimeters
oss_cm = gaps_clean[oss_idx] * 100

print(f"\n  CSS: {css_cm:.2f} cm at {angles[css_idx]}°")
print(f"  OSS: {oss_cm:.2f} cm at {angles[oss_idx]}°")
print(f"\n  Eccentricity: {(oss_cm - css_cm):.2f} cm")
print(f"  Mean gap: {np.nanmean(gaps_clean)*100:.2f} cm")
print("=" * 55)

# ---------------------------------------------------------
# STEP 9: 3D VISUALIZATION
# ---------------------------------------------------------
print("\n--- Generating Visualizations ---")

fig = go.Figure()

# Surface plot of DEM
fig.add_trace(go.Surface(
    x=xi, y=yi, z=zi_smooth,
    colorscale='Viridis', opacity=0.4, showscale=False
))

# Mantle peak marker
fig.add_trace(go.Scatter3d(
    x=[peak_x], y=[peak_y], z=[peak_z],
    mode='markers',
    marker=dict(size=10, color='red', symbol='diamond'),
    name='Mantle Peak'
))

# Raw detection points (semi-transparent)
valid_m = ~np.isnan(mantle_radii)
raw_mx = peak_x + mantle_radii[valid_m] * np.cos(np.radians(angles[valid_m]))
raw_my = peak_y + mantle_radii[valid_m] * np.sin(np.radians(angles[valid_m]))
fig.add_trace(go.Scatter3d(
    x=raw_mx, y=raw_my, z=np.full(len(raw_mx), z_measure),
    mode='markers', marker=dict(size=2, color='lightgreen', opacity=0.4),
    name='Mantle Raw'
))

valid_w = ~np.isnan(wall_radii)
raw_wx = peak_x + wall_radii[valid_w] * np.cos(np.radians(angles[valid_w]))
raw_wy = peak_y + wall_radii[valid_w] * np.sin(np.radians(angles[valid_w]))
fig.add_trace(go.Scatter3d(
    x=raw_wx, y=raw_wy, z=np.full(len(raw_wx), z_measure),
    mode='markers', marker=dict(size=2, color='lightblue', opacity=0.4),
    name='Wall Raw'
))

# Smoothed contour lines
rads = np.radians(angles)
mantle_x = peak_x + mantle_smooth * np.cos(rads)
mantle_y = peak_y + mantle_smooth * np.sin(rads)
wall_x = peak_x + wall_smooth * np.cos(rads)
wall_y = peak_y + wall_smooth * np.sin(rads)

fig.add_trace(go.Scatter3d(
    x=mantle_x, y=mantle_y, z=np.full(360, z_measure),
    mode='lines', line=dict(color='lime', width=10),
    name='Mantle (Smoothed)'
))

fig.add_trace(go.Scatter3d(
    x=wall_x, y=wall_y, z=np.full(360, z_measure),
    mode='lines', line=dict(color='dodgerblue', width=10),
    name=f'Wall (Height>{WALL_HEIGHT_THRESHOLD}m)'
))

# CSS and OSS indicator lines
fig.add_trace(go.Scatter3d(
    x=[mantle_x[css_idx], wall_x[css_idx]],
    y=[mantle_y[css_idx], wall_y[css_idx]],
    z=[z_measure, z_measure],
    mode='lines+markers',
    line=dict(color='red', width=12),
    marker=dict(size=8),
    name=f'CSS: {css_cm:.2f} cm'
))

fig.add_trace(go.Scatter3d(
    x=[mantle_x[oss_idx], wall_x[oss_idx]],
    y=[mantle_y[oss_idx], wall_y[oss_idx]],
    z=[z_measure, z_measure],
    mode='lines+markers',
    line=dict(color='orange', width=12),
    marker=dict(size=8),
    name=f'OSS: {oss_cm:.2f} cm'
))

# Gap lines every 15 degrees
for i in range(0, 360, 15):
    if not np.isnan(gaps_clean[i]):
        fig.add_trace(go.Scatter3d(
            x=[mantle_x[i], wall_x[i]],
            y=[mantle_y[i], wall_y[i]],
            z=[z_measure, z_measure],
            mode='lines', line=dict(color='gray', width=2),
            opacity=0.4, showlegend=False,
            hovertext=f'{i}°: {gaps_clean[i]*100:.2f} cm'
        ))

fig.update_layout(
    title=f"Crusher Gap Analysis (v9 - Height>{WALL_HEIGHT_THRESHOLD}m)<br>" +
          f"CSS: {css_cm:.2f} cm | OSS: {oss_cm:.2f} cm",
    scene=dict(
        aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    width=1100, height=850,
    legend=dict(x=0.02, y=0.98)
)

fig.show()

# ---------------------------------------------------------
# DIAGNOSTIC: Radial Height Profile
# ---------------------------------------------------------
fig4 = go.Figure()

for angle_deg in [angles[css_idx], angles[oss_idx], 0, 90, 180, 270]:
    theta = np.radians(angle_deg)
    r_samples = np.linspace(0.01, 0.30, 200)
    x_ray = peak_x + np.cos(theta) * r_samples
    y_ray = peak_y + np.sin(theta) * r_samples
    z_ray = interp_z(np.c_[y_ray, x_ray])
    
    color = 'red' if angle_deg == angles[css_idx] else \
            ('orange' if angle_deg == angles[oss_idx] else 'gray')
    width = 3 if angle_deg in [angles[css_idx], angles[oss_idx]] else 1
    
    fig4.add_trace(go.Scatter(
        x=r_samples * 100, y=z_ray * 100,
        mode='lines', line=dict(color=color, width=width),
        name=f'{angle_deg}°'
    ))

# Add threshold and floor reference lines
fig4.add_hline(y=WALL_HEIGHT_THRESHOLD*100, line_dash='dash', line_color='blue', 
               annotation_text=f'Wall threshold ({WALL_HEIGHT_THRESHOLD*100:.0f}cm)')
fig4.add_hline(y=floor_z*100, line_dash='dash', line_color='brown', 
               annotation_text='Floor')

fig4.update_layout(
    title="Radial Height Profile<br>" +
          "<span style='font-size:12px'>Wall detected where height crosses the blue threshold</span>",
    xaxis_title="Distance from peak (cm)",
    yaxis_title="Height (cm)",
    width=900, height=500
)
fig4.show()

# ---------------------------------------------------------
# POLAR PLOTS
# ---------------------------------------------------------
# Contour polar plot
fig2 = go.Figure()
fig2.add_trace(go.Scatterpolar(
    r=mantle_smooth * 100, theta=angles, mode='lines', 
    line=dict(color='green', width=3), name='Mantle'
))
fig2.add_trace(go.Scatterpolar(
    r=wall_smooth * 100, theta=angles, mode='lines', 
    line=dict(color='blue', width=3), name='Wall'
))
fig2.update_layout(
    title="Contours (Polar View)",
    polar=dict(radialaxis=dict(range=[0, 25])),
    width=700, height=700
)
fig2.show()

# Gap distribution polar plot
fig3 = go.Figure()
fig3.add_trace(go.Scatterpolar(
    r=gaps_clean * 100, theta=angles, mode='lines', 
    line=dict(color='blue', width=2), fill='toself', 
    fillcolor='rgba(0,0,255,0.1)', name='Gap'
))
fig3.add_trace(go.Scatterpolar(
    r=[css_cm], theta=[angles[css_idx]], mode='markers',
    marker=dict(size=15, color='red', symbol='star'),
    name=f'CSS: {css_cm:.1f}cm'
))
fig3.add_trace(go.Scatterpolar(
    r=[oss_cm], theta=[angles[oss_idx]], mode='markers',
    marker=dict(size=15, color='orange', symbol='star'),
    name=f'OSS: {oss_cm:.1f}cm'
))
fig3.update_layout(
    title="Gap Distribution (Polar View)",
    polar=dict(radialaxis=dict(range=[0, max(oss_cm*1.2, 12)])),
    width=700, height=700
)
fig3.show()

# ---------------------------------------------------------
# COMPLETION MESSAGE
# ---------------------------------------------------------
print("\n✓ Analysis Complete!")
print(f"\nConfiguration:")
print(f"  Wall Height Threshold: {WALL_HEIGHT_THRESHOLD}m ({WALL_HEIGHT_THRESHOLD*100:.0f}cm)")
print(f"  - Lower value = wall detected closer to mantle")
print(f"  - Higher value = wall detected further from mantle")
