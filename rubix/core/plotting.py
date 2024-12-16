import h5py
import matplotlib.pyplot as plt


def stellar_age_histogram(h5_file):
    with h5py.File(h5_file, "r") as f:
        star_ages = f["particles/stars/age"][:]
    plt.figure(figsize=(8, 6))
    plt.hist(star_ages, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
    plt.xlabel("age [Gyr]")
    plt.ylabel("number of stars")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
