"""
Batch CSS/OSS Analyzer 
==========================================

This script processes multiple LiDAR CSV files to measure CSS and OSS values,
generates interactive 3D visualizations for each file, and creates a unified
dashboard for navigating all results.

Features:
    - Batch processing of multiple Test*.csv files
    - Interactive 3D visualization for each scan (HTML output)
    - Unified dashboard with search, navigation, and statistics
    - CSV export of all measurements

Usage:
    1. Place all Test*.csv files in the same directory as this script
    2. Run: python batch_analyzer2.py
    3. Open visualizations/index.html in a web browser

Output:
    - visualizations/index.html         : Interactive dashboard
    - visualizations/Test*_visualization.html : Individual 3D views
    - batch_analysis_results_v9.csv     : CSV with all measurements

Algorithm:
    Uses the same v9 algorithm as test2.py:
    - Absolute Height Wall Detection
    - Automatic mantle peak detection
    - Circle fitting + Savitzky-Golay smoothing

Dependencies:
    - pandas
    - numpy
    - scipy
    - plotly
    - os
    - sys

Author: Behlah Katleriwala, Yassh Singh, Mayank Mehan, Anjali Singh
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter, maximum_filter, sobel
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import sys

# Configure stdout encoding for cross-platform compatibility
sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Wall Height Threshold (meters)
# The wall boundary is detected where surface height exceeds this value
WALL_HEIGHT_THRESHOLD = 0.35  # 35 cm - adjust based on crusher geometry


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def rotate_y_90(points):
    """
    Rotate point cloud 90 degrees around the Y-axis for axis alignment.
    
    Args:
        points: numpy array of shape (N, 3) with XYZ coordinates
        
    Returns:
        Rotated points array
    """
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    return points @ R.T


def fit_circle_to_radii(radii, angles_deg, peak_x, peak_y):
    """
    Fit a circle to radial measurements for contour smoothing.
    
    Uses algebraic least-squares circle fitting to create smooth,
    circular contours from noisy radial measurements.
    
    Args:
        radii: Array of 360 radial distances (may contain NaN)
        angles_deg: Array of angles (0-359 degrees)
        peak_x, peak_y: Center point coordinates
        
    Returns:
        fitted_radii: Smoothed radii based on fitted circle
        circle_params: Tuple (cx, cy, r) - center and radius
    """
    valid = ~np.isnan(radii)
    if np.sum(valid) < 10:
        return radii, (peak_x, peak_y, np.nanmean(radii))
    
    # Convert to Cartesian coordinates
    rads = np.radians(angles_deg[valid])
    x_pts = peak_x + radii[valid] * np.cos(rads)
    y_pts = peak_y + radii[valid] * np.sin(rads)
    
    # Algebraic circle fitting: Ax = b
    A = np.column_stack([x_pts, y_pts, np.ones_like(x_pts)])
    b = x_pts**2 + y_pts**2
    
    try:
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        cx = result[0] / 2
        cy = result[1] / 2
        r = np.sqrt(result[2] + cx**2 + cy**2)
    except:
        cx, cy = np.mean(x_pts), np.mean(y_pts)
        r = np.mean(np.sqrt((x_pts - cx)**2 + (y_pts - cy)**2))
    
    # Calculate fitted radius at each angle
    all_rads = np.radians(angles_deg)
    fitted_radii = []
    
    for theta in all_rads:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = peak_x - cx
        dy = peak_y - cy
        
        # Solve quadratic for ray-circle intersection
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


def detect_mantle_peak(xi, yi, zi_smooth):
    """
    Automatically detect the mantle peak (cone apex).
    
    Finds the highest point that is also near the center of the scan,
    using a weighted scoring system (70% height, 30% centrality).
    
    Args:
        xi, yi: Meshgrid coordinate arrays
        zi_smooth: Smoothed height values (DEM)
        
    Returns:
        Tuple (peak_x, peak_y, peak_z): Peak coordinates
    """
    # Find local maxima
    local_max = maximum_filter(zi_smooth, size=5)
    peaks_mask = (zi_smooth == local_max) & (zi_smooth > 0)
    peak_indices = np.argwhere(peaks_mask)
    
    if len(peak_indices) == 0:
        # Fallback to global maximum
        idx = np.unravel_index(np.nanargmax(zi_smooth), zi_smooth.shape)
        return xi[idx], yi[idx], zi_smooth[idx]

    # Score candidates by height and centrality
    center_x = (xi.min() + xi.max()) / 2
    center_y = (yi.min() + yi.max()) / 2
    best_score, best_peak = -1, None
    
    for idx in peak_indices:
        px = xi[idx[0], idx[1]]
        py = yi[idx[0], idx[1]]
        pz = zi_smooth[idx[0], idx[1]]
        
        dist = np.sqrt((px - center_x)**2 + (py - center_y)**2)
        height_norm = pz / zi_smooth.max()
        dist_norm = 1 - (dist / np.sqrt((xi.max()-xi.min())**2 + (yi.max()-yi.min())**2))
        score = 0.7 * height_norm + 0.3 * dist_norm
        
        if score > best_score:
            best_score = score
            best_peak = (px, py, pz)
            
    return best_peak


# =============================================================================
# VISUALIZATION GENERATION
# =============================================================================

def create_visualization(filename, xi, yi, zi_smooth, peak_x, peak_y, peak_z,
                        z_measure, mantle_smooth, wall_smooth, angles,
                        gaps_clean, css_cm, oss_cm, css_idx, oss_idx,
                        mantle_radii, wall_radii, output_dir="visualizations"):
    """
    Create an interactive 3D visualization and save as HTML.
    
    Generates a Plotly figure with:
    - Surface plot of the DEM
    - Mantle and wall contours
    - CSS and OSS indicator lines
    - Gap measurement lines
    
    Args:
        filename: Source CSV filename (for title)
        xi, yi, zi_smooth: DEM grid data
        peak_x, peak_y, peak_z: Mantle peak coordinates
        z_measure: Height of measurement plane
        mantle_smooth, wall_smooth: Smoothed contour radii
        angles: Array of angles (0-359)
        gaps_clean: Gap values at each angle
        css_cm, oss_cm: CSS and OSS values in centimeters
        css_idx, oss_idx: Indices of CSS and OSS angles
        mantle_radii, wall_radii: Raw radii (for reference)
        output_dir: Directory for HTML output
        
    Returns:
        Path to the saved HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig = go.Figure()
    
    # DEM surface
    fig.add_trace(go.Surface(
        x=xi, y=yi, z=zi_smooth,
        colorscale='Viridis', opacity=0.4, showscale=False
    ))
    
    # Peak marker
    fig.add_trace(go.Scatter3d(
        x=[peak_x], y=[peak_y], z=[peak_z],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Mantle Peak'
    ))
    
    # Convert radii to Cartesian coordinates for plotting
    rads = np.radians(angles)
    mantle_x = peak_x + mantle_smooth * np.cos(rads)
    mantle_y = peak_y + mantle_smooth * np.sin(rads)
    wall_x = peak_x + wall_smooth * np.cos(rads)
    wall_y = peak_y + wall_smooth * np.sin(rads)
    
    # Mantle contour (green)
    fig.add_trace(go.Scatter3d(
        x=mantle_x, y=mantle_y, z=np.full(360, z_measure),
        mode='lines', line=dict(color='lime', width=8),
        name='Mantle'
    ))
    
    # Wall contour (blue)
    fig.add_trace(go.Scatter3d(
        x=wall_x, y=wall_y, z=np.full(360, z_measure),
        mode='lines', line=dict(color='dodgerblue', width=8),
        name='Wall'
    ))
    
    # CSS indicator line (blue)
    fig.add_trace(go.Scatter3d(
        x=[mantle_x[css_idx], wall_x[css_idx]],
        y=[mantle_y[css_idx], wall_y[css_idx]],
        z=[z_measure, z_measure],
        mode='lines+markers',
        line=dict(color='blue', width=12),
        marker=dict(size=8, color='blue'),
        name=f'CSS: {css_cm:.2f} cm'
    ))
    
    # OSS indicator line (red)
    fig.add_trace(go.Scatter3d(
        x=[mantle_x[oss_idx], wall_x[oss_idx]],
        y=[mantle_y[oss_idx], wall_y[oss_idx]],
        z=[z_measure, z_measure],
        mode='lines+markers',
        line=dict(color='red', width=12),
        marker=dict(size=8, color='red'),
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
        title=f"{filename}<br>CSS: {css_cm:.2f} cm | OSS: {oss_cm:.2f} cm",
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        width=1000, height=800,
        legend=dict(x=0.02, y=0.98)
    )
    
    # Save as HTML
    base_name = os.path.splitext(filename)[0]
    html_path = os.path.join(output_dir, f"{base_name}_visualization.html")
    fig.write_html(html_path)
    
    return html_path


# =============================================================================
# SINGLE FILE PROCESSING
# =============================================================================

def process_csv_file(csv_path, verbose=False, generate_viz=True, output_dir="visualizations"):
    """
    Process a single CSV file using the algorithm.
    
    Implements the complete CSS/OSS measurement pipeline:
    1. Load and transform point cloud
    2. Build DEM and detect peak
    3. Detect mantle edge and wall boundary
    4. Apply smoothing
    5. Calculate CSS/OSS
    6. Generate visualization (optional)
    
    Args:
        csv_path: Path to the CSV file
        verbose: Print detailed progress if True
        generate_viz: Generate HTML visualization if True
        output_dir: Directory for visualization output
        
    Returns:
        Dictionary with results, or None if processing failed:
        {
            'filename': str,
            'num_points': int,
            'css': float (cm),
            'oss': float (cm),
            'mean_gap': float (cm),
            'std_gap': float (cm),
            'eccentricity': float (cm),
            'coverage': float (%),
            'peak_height': float (cm),
            'floor_z': float (cm),
            'visualization': str (path)
        }
    """
    try:
        filename = os.path.basename(csv_path)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            print(f"{'='*60}")
        
        # -------------------------------------------------------
        # STEP 1: Load and validate data
        # -------------------------------------------------------
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Verify required columns exist
        if not all(k in df.columns for k in ['x', 'y', 'z']):
            if verbose:
                print(f"  ❌ Missing X, Y, Z columns - SKIPPED")
            return None
        
        points = df[['x', 'y', 'z']].astype(float).values
        
        # Require minimum number of points
        if len(points) < 1000:
            if verbose:
                print(f"  ❌ Insufficient points ({len(points)}) - SKIPPED")
            return None
        
        # -------------------------------------------------------
        # STEP 2: Axis alignment
        # -------------------------------------------------------
        points = rotate_y_90(points)
        points[:, 0] -= np.min(points[:, 0])
        points[:, 2] -= np.min(points[:, 2])
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        if verbose:
            print(f"  ✓ Loaded {len(points)} points")
        
        # -------------------------------------------------------
        # STEP 3: Build DEM
        # -------------------------------------------------------
        grid_res, sigma = 400, 2
        xi_arr = np.linspace(x.min(), x.max(), grid_res)
        yi_arr = np.linspace(y.min(), y.max(), grid_res)
        xi, yi = np.meshgrid(xi_arr, yi_arr)
        zi = griddata((x, y), z, (xi, yi), method="linear", fill_value=0)
        zi_smooth = gaussian_filter(zi, sigma=sigma)
        
        # Compute gradients for mantle edge detection
        gx = sobel(zi_smooth, axis=1)
        gy = sobel(zi_smooth, axis=0)
        grad_mag = np.hypot(gx, gy)
        
        # Create interpolators
        interp_z = RegularGridInterpolator(
            (yi_arr, xi_arr), zi_smooth,
            bounds_error=False, fill_value=np.nan
        )
        interp_grad = RegularGridInterpolator(
            (yi_arr, xi_arr), grad_mag,
            bounds_error=False, fill_value=0.0
        )
        
        # -------------------------------------------------------
        # STEP 4: Detect mantle peak
        # -------------------------------------------------------
        peak_x, peak_y, peak_z = detect_mantle_peak(xi, yi, zi_smooth)
        
        if verbose:
            print(f"  ✓ Peak at ({peak_x:.3f}, {peak_y:.3f}, {peak_z:.3f}) m")
        
        # -------------------------------------------------------
        # STEP 5: Detect floor and set measurement plane
        # -------------------------------------------------------
        hist, bin_edges = np.histogram(z, bins=100)
        floor_idx = np.argmax(hist)
        floor_z = (bin_edges[floor_idx] + bin_edges[floor_idx + 1]) / 2
        
        z_measure = floor_z + 0.02
        plane_tol = 0.015
        
        # -------------------------------------------------------
        # STEP 6: Mantle edge detection
        # -------------------------------------------------------
        angles = np.arange(0, 360, 1)
        num_samples = 200
        max_search_radius = 0.20
        
        mantle_radii = []
        
        for angle_deg in angles:
            theta = np.radians(angle_deg)
            r_samples = np.linspace(0.01, max_search_radius, num_samples)
            x_ray = peak_x + np.cos(theta) * r_samples
            y_ray = peak_y + np.sin(theta) * r_samples
            
            z_ray = interp_z(np.c_[y_ray, x_ray])
            grad_ray = interp_grad(np.c_[y_ray, x_ray])
            
            mantle_mask = (z_ray >= floor_z + 0.005) & (z_ray <= z_measure + plane_tol * 2)
            
            if np.any(mantle_mask):
                valid_indices = np.where(mantle_mask)[0]
                
                if len(valid_indices) > 5:
                    grad_in_region = grad_ray[valid_indices]
                    grad_threshold = np.percentile(grad_in_region, 70)
                    high_grad = grad_in_region > grad_threshold
                    
                    if np.any(high_grad):
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
        
        # -------------------------------------------------------
        # STEP 7: Wall detection (absolute height threshold)
        # -------------------------------------------------------
        wall_radii = []
        
        for angle_deg in angles:
            theta = np.radians(angle_deg)
            mantle_r = mantle_radii[angle_deg] if not np.isnan(mantle_radii[angle_deg]) else 0.05
            
            r_samples = np.linspace(mantle_r + 0.01, 0.35, 200)
            x_ray = peak_x + np.cos(theta) * r_samples
            y_ray = peak_y + np.sin(theta) * r_samples
            
            z_ray = interp_z(np.c_[y_ray, x_ray])
            
            # Wall = first point exceeding height threshold
            above_wall_height = z_ray > WALL_HEIGHT_THRESHOLD
            
            if np.any(above_wall_height):
                first_wall_idx = np.where(above_wall_height)[0][0]
                wall_r = r_samples[first_wall_idx]
                wall_radii.append(wall_r)
            else:
                wall_radii.append(np.nan)
        
        wall_radii = np.array(wall_radii)
        
        # -------------------------------------------------------
        # STEP 8: Contour smoothing
        # -------------------------------------------------------
        mantle_fitted, mantle_circle = fit_circle_to_radii(mantle_radii, angles, peak_x, peak_y)
        wall_fitted, wall_circle = fit_circle_to_radii(wall_radii, angles, peak_x, peak_y)
        
        # Interpolate NaN values
        mantle_interp = pd.Series(mantle_radii).interpolate(limit_direction='both')
        mantle_interp = mantle_interp.fillna(np.nanmedian(mantle_radii))
        wall_interp = pd.Series(wall_radii).interpolate(limit_direction='both')
        wall_interp = wall_interp.fillna(np.nanmedian(wall_radii))
        
        # Savitzky-Golay smoothing
        mantle_savgol = savgol_filter(mantle_interp, 51, 3)
        wall_savgol = savgol_filter(wall_interp, 51, 3)
        
        # Blend: 70% circle fit + 30% Savgol
        blend = 0.7
        mantle_smooth = blend * mantle_fitted + (1 - blend) * mantle_savgol
        wall_smooth = blend * wall_fitted + (1 - blend) * wall_savgol
        
        # -------------------------------------------------------
        # STEP 9: CSS/OSS calculation
        # -------------------------------------------------------
        gaps = wall_smooth - mantle_smooth
        
        valid_gaps = gaps > 0
        if not np.all(valid_gaps):
            gaps_clean = np.where(valid_gaps, gaps, np.nan)
        else:
            gaps_clean = gaps
        
        if np.all(np.isnan(gaps_clean)):
            if verbose:
                print(f"  ❌ No valid gaps - SKIPPED")
            return None
        
        css_idx = np.nanargmin(gaps_clean)
        oss_idx = np.nanargmax(gaps_clean)
        
        css_cm = gaps_clean[css_idx] * 100
        oss_cm = gaps_clean[oss_idx] * 100
        mean_gap = np.nanmean(gaps_clean) * 100
        std_gap = np.nanstd(gaps_clean) * 100
        eccentricity = oss_cm - css_cm
        
        valid_wall_count = np.sum(~np.isnan(wall_radii))
        coverage = valid_wall_count / 360 * 100
        
        if verbose:
            print(f"  ✓ CSS: {css_cm:.2f} cm | OSS: {oss_cm:.2f} cm | Coverage: {coverage:.1f}%")
        
        # -------------------------------------------------------
        # STEP 10: Generate visualization
        # -------------------------------------------------------
        viz_path = None
        if generate_viz:
            viz_path = create_visualization(
                filename, xi, yi, zi_smooth, peak_x, peak_y, peak_z,
                z_measure, mantle_smooth, wall_smooth, angles,
                gaps_clean, css_cm, oss_cm, css_idx, oss_idx,
                mantle_radii, wall_radii, output_dir
            )
            if verbose:
                print(f"  ✓ Visualization saved: {viz_path}")
        
        return {
            'filename': filename,
            'num_points': len(points),
            'css': css_cm,
            'oss': oss_cm,
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'eccentricity': eccentricity,
            'coverage': coverage,
            'peak_height': peak_z * 100,
            'floor_z': floor_z * 100,
            'visualization': viz_path
        }
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# DASHBOARD GENERATION
# =============================================================================

def create_dashboard(results, output_dir="visualizations"):
    """
    Create an interactive HTML dashboard for viewing all results.
    
    Features:
    - Summary statistics (avg CSS, OSS, total points)
    - Searchable file list
    - Interactive 3D visualization viewer
    - Keyboard navigation (arrow keys)
    
    Args:
        results: List of result dictionaries from process_csv_file()
        output_dir: Directory for HTML output
        
    Returns:
        Path to the dashboard HTML file
    """
    # Convert results to JSON-serializable format
    js_data = []
    for r in results:
        js_data.append({
            'name': r['filename'].replace('.csv', ''),
            'css': float(round(r['css'], 2)),
            'oss': float(round(r['oss'], 2)),
            'points': int(r['num_points']),
            'coverage': float(round(r['coverage'], 1)),
            'eccentricity': float(round(r['eccentricity'], 2))
        })
    
    # Calculate summary statistics
    avg_css = np.mean([r['css'] for r in results])
    avg_oss = np.mean([r['oss'] for r in results])
    total_points = sum(r['num_points'] for r in results)
    
    # Generate HTML with embedded CSS and JavaScript
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS/OSS Analysis Dashboard v9</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Reset and base styles */
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        :root {{
            --bg-dark: #0a0e17;
            --bg-card: #111827;
            --bg-hover: #1f2937;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --accent-purple: #8b5cf6;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --border-color: #374151;
        }}

        body {{
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at top left, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(6, 182, 212, 0.1) 0%, transparent 50%);
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, var(--bg-card) 0%, rgba(17, 24, 39, 0.95) 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 1.5rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(10px);
        }}

        .header h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .header p {{
            color: var(--text-secondary);
            margin-top: 0.25rem;
            font-size: 0.9rem;
        }}

        .container {{ max-width: 1600px; margin: 0 auto; padding: 2rem; }}

        /* Stats grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-purple);
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
        }}

        .stat-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}

        .stat-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--accent-cyan);
        }}

        .stat-value.green {{ color: var(--accent-green); }}
        .stat-value.blue {{ color: var(--accent-blue); }}
        .stat-value.yellow {{ color: var(--accent-yellow); }}
        .stat-value.purple {{ color: var(--accent-purple); }}

        /* Main layout */
        .main-content {{
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 1.5rem;
            height: calc(100vh - 280px);
            min-height: 600px;
        }}

        /* Sidebar */
        .sidebar {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        .sidebar-header {{
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            background: rgba(139, 92, 246, 0.05);
        }}

        .sidebar-header h2 {{ font-size: 0.9rem; font-weight: 600; }}

        .file-list {{
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
        }}

        .file-list::-webkit-scrollbar {{ width: 6px; }}
        .file-list::-webkit-scrollbar-track {{ background: var(--bg-dark); }}
        .file-list::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px; }}

        .file-item {{
            display: flex;
            flex-direction: column;
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }}

        .file-item:hover {{
            background: var(--bg-hover);
            border-color: var(--border-color);
        }}

        .file-item.active {{
            background: rgba(139, 92, 246, 0.15);
            border-color: var(--accent-purple);
        }}

        .file-name {{ font-weight: 500; font-size: 0.9rem; margin-bottom: 0.25rem; }}

        .file-stats {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-secondary);
            display: flex;
            gap: 1rem;
        }}

        .file-css {{ color: var(--accent-blue); }}
        .file-oss {{ color: var(--accent-red); }}

        /* Viewer panel */
        .viewer-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        .viewer-header {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(6, 182, 212, 0.05);
        }}

        .viewer-title {{ font-size: 1.1rem; font-weight: 600; }}

        .viewer-metrics {{ display: flex; gap: 1.5rem; }}

        .metric {{ display: flex; align-items: center; gap: 0.5rem; }}

        .metric-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .metric-dot.css {{ background: var(--accent-blue); }}
        .metric-dot.oss {{ background: var(--accent-red); }}

        .metric-label {{ font-size: 0.8rem; color: var(--text-secondary); }}
        .metric-value {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; }}

        .viewer-content {{ flex: 1; position: relative; }}
        .viewer-content iframe {{ width: 100%; height: 100%; border: none; background: #fff; }}

        .placeholder {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
        }}

        .placeholder-icon {{ font-size: 4rem; margin-bottom: 1rem; opacity: 0.3; }}

        /* Navigation */
        .nav-buttons {{ display: flex; gap: 0.5rem; }}

        .nav-btn {{
            background: var(--bg-hover);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }}

        .nav-btn:hover {{ background: var(--accent-purple); border-color: var(--accent-purple); }}
        .nav-btn:disabled {{ opacity: 0.3; cursor: not-allowed; }}

        /* Search */
        .search-box {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); }}

        .search-input {{
            width: 100%;
            padding: 0.6rem 1rem;
            background: var(--bg-dark);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
        }}

        .search-input:focus {{ outline: none; border-color: var(--accent-purple); }}
        .search-input::placeholder {{ color: var(--text-secondary); }}

        .version-badge {{
            background: var(--accent-purple);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            margin-left: 0.5rem;
        }}

        /* Responsive */
        @media (max-width: 1024px) {{
            .main-content {{ grid-template-columns: 1fr; height: auto; }}
            .sidebar {{ max-height: 300px; }}
            .viewer-panel {{ min-height: 500px; }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <h1>CSS/OSS Analysis Dashboard <span class="version-badge">v9</span></h1>
        <p>Absolute Height Wall Detection - Interactive 3D Visualization</p>
    </header>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Files</div>
                <div class="stat-value purple">{len(results)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average CSS</div>
                <div class="stat-value blue">{avg_css:.2f} cm</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average OSS</div>
                <div class="stat-value green">{avg_oss:.2f} cm</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Points</div>
                <div class="stat-value yellow">{total_points:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Wall Threshold</div>
                <div class="stat-value">{WALL_HEIGHT_THRESHOLD*100:.0f} cm</div>
            </div>
        </div>

        <div class="main-content">
            <div class="sidebar">
                <div class="sidebar-header">
                    <h2>Test Files</h2>
                </div>
                <div class="search-box">
                    <input type="text" class="search-input" id="searchInput" placeholder="Search files...">
                </div>
                <div class="file-list" id="fileList"></div>
            </div>

            <div class="viewer-panel">
                <div class="viewer-header">
                    <div class="viewer-title" id="viewerTitle">Select a file to view</div>
                    <div class="viewer-metrics" id="viewerMetrics" style="display: none;">
                        <div class="metric">
                            <span class="metric-dot css"></span>
                            <span class="metric-label">CSS:</span>
                            <span class="metric-value" id="currentCss">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-dot oss"></span>
                            <span class="metric-label">OSS:</span>
                            <span class="metric-value" id="currentOss">-</span>
                        </div>
                    </div>
                    <div class="nav-buttons">
                        <button class="nav-btn" id="prevBtn" disabled>Previous</button>
                        <button class="nav-btn" id="nextBtn" disabled>Next</button>
                    </div>
                </div>
                <div class="viewer-content" id="viewerContent">
                    <div class="placeholder">
                        <div class="placeholder-icon">3D</div>
                        <p>Click on a file to view its 3D visualization</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from batch processing
        const testFiles = {json.dumps(js_data)};
        
        let currentIndex = -1;
        let filteredFiles = [...testFiles];

        // Render file list
        function renderFileList(files) {{
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';

            files.forEach((file, index) => {{
                const div = document.createElement('div');
                div.className = 'file-item' + (testFiles.indexOf(file) === currentIndex ? ' active' : '');
                div.innerHTML = `
                    <span class="file-name">${{file.name}}</span>
                    <span class="file-stats">
                        <span class="file-css">CSS: ${{file.css.toFixed(2)}} cm</span>
                        <span class="file-oss">OSS: ${{file.oss.toFixed(2)}} cm</span>
                    </span>
                `;
                div.onclick = () => loadVisualization(testFiles.indexOf(file));
                fileList.appendChild(div);
            }});
        }}

        // Load visualization iframe
        function loadVisualization(index) {{
            currentIndex = index;
            const file = testFiles[index];
            
            // Update active state
            document.querySelectorAll('.file-item').forEach((item, i) => {{
                item.classList.toggle('active', testFiles.indexOf(filteredFiles[i]) === index);
            }});

            // Update header
            document.getElementById('viewerTitle').textContent = file.name;
            document.getElementById('viewerMetrics').style.display = 'flex';
            document.getElementById('currentCss').textContent = file.css.toFixed(2) + ' cm';
            document.getElementById('currentOss').textContent = file.oss.toFixed(2) + ' cm';

            // Load iframe
            const viewerContent = document.getElementById('viewerContent');
            viewerContent.innerHTML = `<iframe src="${{file.name}}_visualization.html"></iframe>`;

            // Update nav buttons
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === testFiles.length - 1;
        }}

        // Navigation handlers
        document.getElementById('prevBtn').onclick = () => {{
            if (currentIndex > 0) loadVisualization(currentIndex - 1);
        }};

        document.getElementById('nextBtn').onclick = () => {{
            if (currentIndex < testFiles.length - 1) loadVisualization(currentIndex + 1);
        }};

        // Search handler
        document.getElementById('searchInput').oninput = (e) => {{
            const query = e.target.value.toLowerCase();
            filteredFiles = testFiles.filter(f => f.name.toLowerCase().includes(query));
            renderFileList(filteredFiles);
        }};

        // Keyboard navigation
        document.onkeydown = (e) => {{
            if (e.key === 'ArrowUp' && currentIndex > 0) {{
                e.preventDefault();
                loadVisualization(currentIndex - 1);
            }} else if (e.key === 'ArrowDown' && currentIndex < testFiles.length - 1) {{
                e.preventDefault();
                loadVisualization(currentIndex + 1);
            }}
        }};

        // Initialize
        renderFileList(testFiles);
        loadVisualization(0);
    </script>
</body>
</html>'''
    
    # Write dashboard file
    dashboard_path = os.path.join(output_dir, "index.html")
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    return dashboard_path


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function for batch processing.
    
    Workflow:
    1. Find all Test*.csv files in current directory
    2. Process each file using algorithm
    3. Generate individual visualizations
    4. Create unified dashboard
    5. Export results to CSV
    """
    print("\n" + "="*70)
    print(" BATCH CSS/OSS ANALYZER")
    print("="*70 + "\n")
    print(f"Wall Height Threshold: {WALL_HEIGHT_THRESHOLD}m ({WALL_HEIGHT_THRESHOLD*100:.0f}cm)\n")
    
    # Find all CSV files matching pattern
    csv_files = sorted(glob.glob("Test*.csv"))
    
    if len(csv_files) == 0:
        print("No Test*.csv files found in current directory!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    # Create output directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Process each file
    results = []
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] {csv_file}...", end=" ")
        result = process_csv_file(csv_file, verbose=False, generate_viz=True, output_dir=viz_dir)
        if result:
            results.append(result)
            print(f"CSS: {result['css']:.2f} cm | OSS: {result['oss']:.2f} cm")
        else:
            print("FAILED")
    
    if len(results) == 0:
        print("\nNo valid results!")
        return
    
    # Print summary table
    print("\n" + "="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    print(f"{'File':<15} {'Points':>8} {'CSS':>10} {'OSS':>10} {'Ecc.':>10} {'Coverage':>10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['filename']:<15} {r['num_points']:>8} "
              f"{r['css']:>10.2f} {r['oss']:>10.2f} "
              f"{r['eccentricity']:>10.2f} {r['coverage']:>9.1f}%")
    
    # Calculate final statistics
    css_values = [r['css'] for r in results]
    oss_values = [r['oss'] for r in results]
    
    avg_css = np.mean(css_values)
    avg_oss = np.mean(oss_values)
    std_css = np.std(css_values)
    std_oss = np.std(oss_values)
    
    print("\n" + "="*70)
    print(" FINAL MEASUREMENTS")
    print("="*70)
    print(f"\n  Average CSS: {avg_css:.2f} cm (+/- {std_css:.2f})")
    print(f"  Average OSS: {avg_oss:.2f} cm (+/- {std_oss:.2f})")
    print(f"  CSS Range: {min(css_values):.2f} - {max(css_values):.2f} cm")
    print(f"  OSS Range: {min(oss_values):.2f} - {max(oss_values):.2f} cm")
    
    # Create dashboard
    dashboard_path = create_dashboard(results, viz_dir)
    print(f"\nDashboard created: {dashboard_path}")
    
    # Save CSV results
    results_df = pd.DataFrame(results)
    results_df.to_csv("batch_analysis_results_v9.csv", index=False)
    print(f"Results saved: batch_analysis_results_v9.csv")
    
    print("\n" + "="*70)
    print(" COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
