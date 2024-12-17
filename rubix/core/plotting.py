import h5py
import matplotlib.pyplot as plt

def stellar_age_histogram(h5_file):
    with h5py.File(h5_file, "r") as f:
        star_ages = f["particles/stars/age"][:]
    plt.figure(figsize=(8, 6))
    plt.hist(star_ages, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
    plt.xlabel("Age [Gyr]")
    plt.ylabel("Number of stars")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def star_coords_2D(h5_file):
    with h5py.File(h5_file, "r") as f:
        star_coords = f['particles/stars/coords'][:]
    x = star_coords[:, 0]
    y = star_coords[:, 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=1, alpha=0.5)
    plt.xlabel(r'$x$ [kpc]')
    plt.ylabel(r'$y$ [kpc]') 
    plt.grid(True)
    plt.show()

def star_metallicity_histogram(h5_file):
    with h5py.File(h5_file, "r") as f:
        star_metallicity = f["particles/stars/metallicity"][:]
    plt.figure(figsize=(8, 6))
    plt.hist(star_metallicity, bins=50, color="forestgreen", edgecolor="black", alpha=0.7)
    plt.xlabel("Metallicity")
    plt.ylabel("Number of stars")
    plt.title("Stellar Metallicity Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
