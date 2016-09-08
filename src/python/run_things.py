import click

import experiments.appgrad.e4.hyperband as eaeh

@click.command()
@click.option('--hdf5-path')
@click.option('--k', default=1)
@click.option('--subject')
@click.option('--seconds', default=1)
@click.option('--max-rounds', default=10)
def run_it_all_day(
        hdf5_path,
        k,
        subject,
        seconds,
        max_rounds):

runner = eaeh.run_n_view_online_appgrad_e4_data_hyperband_experiment(
    hdf5_path,
    k,
    subject,
    seconds=seconds,
    max_rounds=max_rounds)

if __name__=='__main__':
    run_it_all_day()
