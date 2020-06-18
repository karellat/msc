import click
import logging
from deep_mri.train.training import run_train


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
def main(config_file):
    logging.basicConfig(level=logging.INFO)
    run_train(config_file)


if __name__ == "__main__":
    main()
