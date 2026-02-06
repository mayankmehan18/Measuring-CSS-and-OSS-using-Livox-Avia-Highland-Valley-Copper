# Measuring CSS and OSS using Livox-Avia  Highland Valley Copper
Python-based CSS/OSS measurement toolkit using Livox Avia LiDAR point clouds. Builds DEM surfaces from raw scans, automatically finds mantle peak and concave wall boundaries, measures minimum and maximum gap around the crusher, and exports per-scan Plotly visualizations and batch result tables.

# Description

This project automates the measurement of Close Side Setting (CSS) and Open Side Setting (OSS) for a gyratory crusher using Livox Avia LiDAR point cloud data. Instead of manual “bucket method” checks, the pipeline processes a LiDAR scan, reconstructs the crusher surfaces, and outputs repeatable digital CSS/OSS values with clear visual evidence.

At a high level, the system:

* Loads raw LiDAR scans (CSV point clouds).

* Aligns and normalizes the scan so the crusher geometry is consistent.

* Builds a smoothed surface model / DEM to reduce noise while preserving shape.

* Detects the mantle peak and estimates the mantle region.

* Detects the concave wall boundary using a height-based threshold approach (designed to reduce floor/noise influence).

* Uses radial sampling (360°) to compute the minimum gap (CSS) and maximum gap (OSS).

* Produces interactive Plotly HTML visualizations (rays, detected boundaries, and surfaces) plus batch measurement reports for multiple scans.

The goal is a practical, scalable foundation for crusher gap verification that is safer, faster to review, and easier to track over time.

# Getting Started
Dependencies

Python 3.8+

Libraries:

* numpy

* pandas

* open3d

* scipy

* plotly

Install:

~~~
pip install numpy pandas open3d scipy plotly
~~~

File Types
~~~
.lvx: Native Livox recording format (viewable in Livox Viewer).

.csv: Raw point cloud export (X, Y, Z + optional intensity/metadata).

.pcd (optional): Standard point cloud format (useful for CloudCompare/Open3D workflows).
~~~

This repo primarily runs on CSV scans.

# Installing

Clone the repository.

Place your LiDAR Test.csv* files in the same folder as the scripts (recommended for batch mode).

# Executing Program
Single Scan (interactive / one-off)

Open script.py and set the input file path (example: Test1.csv).

Run:

~~~
python script.py
~~~

Batch Mode (recommended)

Place all scans as Test*.csv in the script directory.

Run:

~~~
python batch_analyze.py
~~~

Open the dashboard:

visualizations/index.html

# Outputs
~~~
visualizations/index.html — dashboard to browse scans and results

visualizations/Test*_visualization.html — per-scan interactive 3D views

batch_analysis_results_v9.csv — consolidated CSS/OSS measurements + metadata
~~~
# Notes

* Results depend on scan quality, positioning, and surface noise.

* Thresholds and smoothing parameters can be tuned for different crusher geometries.

* This project is a proof of concept and a foundation for future plant-ready improvements.

# Authors 
* Behlah Katleriwala
* Yassh Singh
* Mayank Mehan
* Anjali Singh 

