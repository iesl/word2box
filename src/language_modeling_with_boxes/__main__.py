import click
from .train.__main__ import train

# from .train.__main__ import evaluate


@click.group()
def main():
    """Using boxes for word representations"""
    pass


main.add_command(train, "train")
# main.add_command(evaluate, "evaluate")
