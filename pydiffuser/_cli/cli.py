import click

from pydiffuser._cli.cli_utils import add_model_subcommands


def create():
    @click.group()
    @click.version_option()
    def pydiffuser_cli():
        """Pydiffuser CLI."""

    add_model_subcommands(pydiffuser_cli)

    return pydiffuser_cli


def cli():
    pydiffuser_cli = create()
    pydiffuser_cli()


if __name__ == "__main__":
    cli()
