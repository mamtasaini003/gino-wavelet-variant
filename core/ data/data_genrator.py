"""
generate_donut_poisson_source_txt.py

Generates 31 .txt files (angles 0..360 inclusive) containing points
only inside an annular (donut) domain. The Az column stores the
Poisson source f(x,y) = sin(m*pi*x) * sin(n*pi*y).

Output columns: x y z Ax Ay Az
"""

import numpy as np
import os
from math import pi
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_donut_data():
    # ---------------- User parameters ----------------
    out_dir = "support_files/donut_poisson_txt"
    os.makedirs(out_dir, exist_ok=True)

    nx, ny = 256, 256
    xlims = (-1.0, 1.0)
    ylims = (-1.0, 1.0)

    r_outer = 1.0
    r_inner = 0.4

    m, n = 1, 1
    angles = np.linspace(0.0, 360.0, 31)
    # -------------------------------------------------

    x = np.linspace(xlims[0], xlims[1], nx)
    y = np.linspace(ylims[0], ylims[1], ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    def rotate_coords(X, Y, angle_deg):
        theta = np.radians(angle_deg)
        Xr = X * np.cos(theta) - Y * np.sin(theta)
        Yr = X * np.sin(theta) + Y * np.cos(theta)
        return Xr, Yr

    for angle in angles:
        Xr, Yr = rotate_coords(X, Y, angle)
        Rr = np.sqrt(Xr**2 + Yr**2)
        mask = (Rr >= r_inner) & (Rr <= r_outer)

        f = np.sin(m * pi * Xr) * np.sin(n * pi * Yr)

        X_inside = Xr[mask]
        Y_inside = Yr[mask]
        Z_inside = np.zeros_like(X_inside)
        Ax = np.zeros_like(X_inside)
        Ay = np.zeros_like(X_inside)
        Az = f[mask]

        data = np.column_stack([X_inside, Y_inside, Z_inside, Ax, Ay, Az])
        fname = f"donut_angle_{int(round(angle)):03d}.txt"
        outpath = os.path.join(out_dir, fname)

        np.savetxt(outpath, data, fmt="%.6f %.6f %.1f %.1f %.1f %.6f",
                   header="x y z Ax Ay Az", comments='')

        print(f"Saved {fname}: {len(X_inside)} points (angle {angle:.1f}°)")

    print(f"\nDone — generated {len(angles)} files in '{out_dir}'")

if __name__ == "__main__":
    generate_donut_data()
