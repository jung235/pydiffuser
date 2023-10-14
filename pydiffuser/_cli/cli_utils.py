from pydiffuser.models import CONFIG_REGISTRY, MODEL_REGISTRY


def get_model_info():
    _ = "1d, 2d, 3d"
    dims = {
        "abp": "2d",
        "aoup": _,
        "bm": _,
        "levy": _,
        "rtp": _,
        "smoluchowski": "1d, 2d",
    }
    info = [("NAME", "MODEL", "CONFIG", "DIMENSION")]
    info += [
        (k, MODEL_REGISTRY[k].__name__, CONFIG_REGISTRY[k].__name__, dim)
        for k, dim in dims.items()
    ]
    return info
