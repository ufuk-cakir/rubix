from rubix.cosmology import RubixCosmology


def get_cosmology(config: dict):
    """Get the cosmology from the configuration"""
    if config["cosmology"]["name"].upper() == "PLANCK15":
        from rubix.cosmology import PLANCK15

        return PLANCK15

    elif config["cosmology"]["name"].upper() == "CUSTOM":
        return RubixCosmology(**config["cosmology"]["args"])

    else:
        raise ValueError(
            f"Cosmology {config['cosmology']['name']} not supported. Try PLANCK15 or CUSTOM."
        )
