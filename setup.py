from distutils.core import setup

packages = [
    'appgrad',
    'ccalin',
    'genelink',
    'gcca',
    'testers',
    'utils']

setup(
    name='dogcca',
    version='0.01',
    packages=['dogcca.' + p for p in packages])
