import click

from drrobert.file_io import get_timestamped as get_ts
from dogcca.testers.appgrad import CCAProbabilisticModelAppGradTester as CCAPMAGT

@click.command()
@click.option('--num-data', default=1000)
@click.option('--k', default=1)
@click.option('--ds', default='10 20 30')
def run_things_all_day_bb(
    num_data,
    k,
    ds):

    ds = [int(d) for d in ds.split()]
    tester = CCAPMAGT(
        ds,
        k=k,
        num_data=num_data)

    tester.run()

if __name__=='__main__':
    run_things_all_day_bb()
