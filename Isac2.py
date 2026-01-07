import numpy as np
import matplotlib.pyplot as plt

def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title="UE Localization"):
    """
    y_true: [N,2] true (x,y)
    y_pred: [N,2] predicted (x,y)
    """
    err = np.linalg.norm(y_pred - y_true, axis=1)

    plt.figure()
    plt.scatter(y_true[:, 0], y_true[:, 1], marker=".", label="True")
    plt.scatter(y_pred[:, 0], y_pred[:, 1], marker="x", label="Pred")
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    plt.figure()
    plt.hist(err, bins=60)
    plt.title("Position error histogram")
    plt.xlabel("Error (m)")
    plt.ylabel("Count")
    plt.grid(True)

    plt.show()


def plot_trajectory(y_true: np.ndarray, y_pred: np.ndarray, every: int = 1, title="UE Trajectory"):
    """
    Plot as a line (useful when samples are time-ordered).
    """
    yt = y_true[::every]
    yp = y_pred[::every]

    plt.figure()
    plt.plot(yt[:, 0], yt[:, 1], label="True trajectory")
    plt.plot(yp[:, 0], yp[:, 1], label="Pred trajectory")
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_error_heatmap(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 40, title="Error heatmap"):
    """
    Heatmap of average error over space.
    """
    err = np.linalg.norm(y_pred - y_true, axis=1)
    x = y_true[:, 0]
    y = y_true[:, 1]

    # 2D binning
    xedges = np.linspace(x.min(), x.max(), bins + 1)
    yedges = np.linspace(y.min(), y.max(), bins + 1)

    sum_err = np.zeros((bins, bins), dtype=float)
    count = np.zeros((bins, bins), dtype=float)

    xi = np.digitize(x, xedges) - 1
    yi = np.digitize(y, yedges) - 1
    mask = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)

    for a, b, e in zip(xi[mask], yi[mask], err[mask]):
        sum_err[b, a] += e
        count[b, a] += 1

    mean_err = np.divide(sum_err, count, out=np.full_like(sum_err, np.nan), where=count > 0)

    plt.figure()
    plt.imshow(
        mean_err,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar(label="Mean error (m)")
    plt.grid(False)
    plt.show()
