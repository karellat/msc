import click
from deep_mri.train.training import run_train


@click.command()
@click.argument('config', type=click.Path(exists=True))
def run(path_to_config):
    run_train(path_to_config)


if __name__ == "__main__":
    run_train()
