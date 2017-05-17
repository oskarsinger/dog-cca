from distutils.core import setup

packages = [
    'appgrad',
    'ccalin',
    'genelink',
    'gcca',
    'testers',
    'utils']

setup(
    name='ogcca',
    version='0.01',
    packages=['ogcca.' + p for p in packages])
