import click

from pydiffuser._cli.cli_utils import get_model_info


def create():
    @click.group()
    @click.version_option()
    def pydiffuser_cli():
        """Pydiffuser CLI"""

    @pydiffuser_cli.command()
    def list():
        """List all models defined in Pydiffuser."""

        info = get_model_info()
        for name, model, config, dimension in info:
            click.echo(f"{name:16}{model:32}{config:32}{dimension:16}")

    return pydiffuser_cli


def cli():
    pydiffuser_cli = create()
    pydiffuser_cli()


if __name__ == "__main__":
    cli()
