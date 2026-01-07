#!/usr/bin/env python3
"""
Visualize TX/RX on a map (x-y) from a CSV file.

Required columns:
  tx_loc_m_x, tx_loc_m_y, tx_loc_m_z
  rx_loc_m_x, rx_loc_m_y, rx_loc_m_z
  rx_speed

Optional columns (any one):
  time_s, t_idx, timestamp, time, frame

Outputs:
  - 2D map plot: TX + RX path, RX points colored by speed
  - Optional 3D plot (toggle with --plot3d)

Usage:
  python visualize_tx_rx_map.py --csv rays.csv
  python visualize_tx_rx_map.py --csv rays.csv --map floorplan.png --xlim -50 50 --ylim 0 120
  python visualize_tx_rx_map.py --csv rays.csv --plot3d
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_time_column(df: pd.DataFrame):
    candidates = ["time_s", "t_idx", "timestamp", "time", "frame"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--map", default=None, help="Optional background image (floorplan/map) e.g. png/jpg")
    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"))
    ap.add_argument("--title", default="TX/RX Map View (x-y)")
    ap.add_argument("--downsample", type=int, default=1, help="Plot every Nth row for speed (default 1)")
    ap.add_argument("--plot3d", action="store_true", help="Also show a 3D plot (x,y,z)")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    required = [
        "tx_loc_m_x","tx_loc_m_y","tx_loc_m_z",
        "rx_loc_m_x","rx_loc_m_y","rx_loc_m_z",
        "rx_speed",
    ]
    require_cols(df, required)

    # sort by time if present
    tcol = find_time_column(df)
    if tcol:
        df = df.sort_values(tcol).reset_index(drop=True)

    # downsample for plotting if needed
    ds = max(1, int(args.downsample))
    d = df.iloc[::ds].copy()

    tx_x = d["tx_loc_m_x"].to_numpy(dtype=float)
    tx_y = d["tx_loc_m_y"].to_numpy(dtype=float)
    tx_z = d["tx_loc_m_z"].to_numpy(dtype=float)

    rx_x = d["rx_loc_m_x"].to_numpy(dtype=float)
    rx_y = d["rx_loc_m_y"].to_numpy(dtype=float)
    rx_z = d["rx_loc_m_z"].to_numpy(dtype=float)

    rx_speed = d["rx_speed"].to_numpy(dtype=float)

    # TX might be constant; if it moves, we'll plot it as a line too
    tx_is_constant = (np.nanmax(tx_x) - np.nanmin(tx_x) < 1e-6) and (np.nanmax(tx_y) - np.nanmin(tx_y) < 1e-6)

    # ----------------
    # 2D MAP (x-y)
    # ----------------
    plt.figure()
    ax = plt.gca()

    # Optional background map image
    # If you provide --map, you should also provide --xlim and --ylim
    if args.map is not None:
        if args.xlim is None or args.ylim is None:
            raise ValueError("When using --map, please also provide --xlim and --ylim to place the image correctly.")
        img = plt.imread(args.map)
        ax.imshow(img, extent=[args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1]], aspect="auto")

    # RX trajectory line
    ax.plot(rx_x, rx_y, linewidth=1.0, label="RX trajectory")

    # RX points colored by speed
    sc = ax.scatter(rx_x, rx_y, c=rx_speed, s=18)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("rx_speed (m/s)")

    # TX plot
    if tx_is_constant:
        ax.scatter([tx_x[0]], [tx_y[0]], marker="*", s=220, label="TX")
        # annotate TX height
        ax.annotate(f"TX z={tx_z[0]:.2f} m", (tx_x[0], tx_y[0]), textcoords="offset points", xytext=(8, 8))
    else:
        ax.plot(tx_x, tx_y, linewidth=1.5, label="TX path")
        ax.scatter([tx_x[0]], [tx_y[0]], marker="*", s=180, label="TX start")
        ax.scatter([tx_x[-1]], [tx_y[-1]], marker="*", s=180, label="TX end")

    # Annotate RX start/end (and z)
    ax.scatter([rx_x[0]], [rx_y[0]], marker="o", s=80, label="RX start")
    ax.scatter([rx_x[-1]], [rx_y[-1]], marker="s", s=80, label="RX end")
    ax.annotate(f"RX start z={rx_z[0]:.2f} m", (rx_x[0], rx_y[0]), textcoords="offset points", xytext=(8, -14))
    ax.annotate(f"RX end z={rx_z[-1]:.2f} m", (rx_x[-1], rx_y[-1]), textcoords="offset points", xytext=(8, -14))

    ax.set_title(args.title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    ax.axis("equal")

    # Apply bounds if given (useful even without an image)
    if args.xlim is not None:
        ax.set_xlim(args.xlim[0], args.xlim[1])
    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])

    ax.legend()

    # ----------------
    # Optional 3D plot
    # ----------------
    if args.plot3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.plot(rx_x, rx_y, rx_z, label="RX trajectory")
        ax3.scatter(rx_x, rx_y, rx_z, c=rx_speed, s=10)
        if tx_is_constant:
            ax3.scatter([tx_x[0]], [tx_y[0]], [tx_z[0]], marker="*", s=150, label="TX")
        else:
            ax3.plot(tx_x, tx_y, tx_z, label="TX path")
        ax3.set_title("TX/RX 3D View")
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("y (m)")
        ax3.set_zlabel("z (m)")
        ax3.legend()

    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
