import click

from deep_mri.train.training import run_train


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-i", "--int_var", type=click.Tuple([str, int]), multiple=True)
@click.option("-s", "--str_var", type=click.Tuple([str, str]), multiple=True)
@click.option("-f", "--float_var", type=click.Tuple([str, float]), multiple=True)
def main(config_file, int_var, str_var, float_var):
    overriding_config = {**dict(int_var), **dict(str_var), **dict(float_var)}
    run_train(config_file, overriding_config)


if __name__ == "__main__":
    main()
