import click

from pydiffuser.models import CONFIG_REGISTRY, MODEL_REGISTRY


def get_model_info():
    _ = "1d, 2d, 3d"
    dims = {
        "abp": "2d",
        "aoup": _,
        "bm": _,
        "levy": _,
        "mips": _,
        "rtp": _,
        "smoluchowski": "1d, 2d",
        "vicsek": "2d",
    }
    info = [("NAME", "MODEL", "CONFIG", "DIMENSION")]
    info += [
        (k, MODEL_REGISTRY[k].__name__, CONFIG_REGISTRY[k].__name__, dim)
        for k, dim in dims.items()
    ]
    return info


def add_model_subcommands(cli: click.Group) -> None:
    @cli.group(name="model")
    def model_cli():
        """Subcommands related to model."""

    @model_cli.command()
    def list():
        """List all models defined in Pydiffuser."""

        info = get_model_info()  # type: ignore[no-untyped-call]
        for name, model, config, dimension in info:
            click.echo(f"{name:16}{model:32}{config:32}{dimension:16}")
