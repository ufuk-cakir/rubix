import pickle
import numpy as np

# metallicity in units of log10(Z_solar)
# hydrogen number density in units of log10(cm^-3)
# temperature in units of log10(K)

data_dir = "./templates/"
cube_path = f"{data_dir}UVB_plus_CMB_strongUV_line_emissivities.dat"


def load_anything(filepath):
    with open(filepath, "rb") as f:
        to_load = pickle.load(f)
    return to_load


def show_data(datafile_path):
    d = load_anything(datafile_path)
    print(type(d), "keys:", *list(d.keys()), sep="\n", end="\n\n")

    data_key = "line_emissivity"
    data_dim = list(d.keys()).index(data_key)

    for i in list(range(data_dim)):
        print(
            f'key = "{list(d.keys())[i]}"',
            f"type = {type(d[list(d.keys())[i]])}",
            f"shape = {d[list(d.keys())[i]].shape}",
            f"values: {d[list(d.keys())[i]]}",
            sep="\n",
            end="\n\n",
        )

    i = list(d.keys()).index("line_name")
    print(
        f'key = "{list(d.keys())[i]}"',
        f"type = {type(d[list(d.keys())[i]])}",
        f"len = {len(d[list(d.keys())[i]])}",
        f"values:",
        *d[list(d.keys())[i]],
        sep="\n",
        end="\n\n",
    )

    print(
        f'Key of data: "{data_key}"',
    )
    print(type(d[data_key]), f"len = {len(d[data_key])} (= number of lines)", sep="\n")
    print(
        f"The list contains for each line a",
        f"{type(d[data_key][0])}",
        f"with shape = {d[data_key][0].shape}",
        f"and data of the type {d[data_key][0].dtype}",
        sep="\n",
        end="\n\n\n",
    )


def load_data(datafile_path):
    d = load_anything(datafile_path)
    data_key = "line_emissivity"
    data_dim = list(d.keys()).index(data_key)
    y_grid = d[data_key]
    x_grid_vectors = [d[k] for k in list(d.keys())[:data_dim]]
    return [x_grid_vectors, y_grid]


def main():
    show_data(cube_path)
    x_grid_vectors, y_grid = load_data(cube_path)
    # print(x_grid_vectors,len(y_grid),y_grid[0].shape,sep='\n')


main()


test = load_data(cube_path)


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def get_emission_flux(
    datafile_path, redshift_value, metallicity_value, hden_value, temp_value
):
    x_grid_vectors, y_grid = load_data(datafile_path)

    redshift_idx = find_nearest(x_grid_vectors[0], redshift_value)
    metallicity_idx = find_nearest(x_grid_vectors[1], metallicity_value)
    hden_idx = find_nearest(x_grid_vectors[2], hden_value)
    temp_idx = find_nearest(x_grid_vectors[3], temp_value)

    emission_flux = []
    for line_data in y_grid:
        flux = line_data[redshift_idx, metallicity_idx, hden_idx, temp_idx]
        emission_flux.append(flux)

    return emission_flux


def get_line_names(datafile_path):
    data = load_anything(datafile_path)
    return data["line_name"]


# Example usage:
redshift_value = 3.0
metallicity_value = -1.0
hden_value = 0.0
temp_value = 4.0

flux_values = get_emission_flux(
    cube_path, redshift_value, metallicity_value, hden_value, temp_value
)
line_names = get_line_names(cube_path)

for line_name, flux in zip(line_names, flux_values):
    print(f"Line: {line_name}, Flux: {flux}")
