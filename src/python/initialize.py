import os

from subprocess import call

def augment_python_path(base, names):

    for name in names:

        full = os.path.join(base, name)

        if os.path.isdir(full):
            call([
                "export",
                "PYTHONPATH=\"${PYTHONPATH}:" + full + "\""])

def add_modules(codebase_path):

    cwd = os.getcwd()
    cwd_modules = os.listdir(cwd)
    codebase_modules = os.listdir(codebase_path)

    augment_python_path(cwd, cwd_modules)
    augment_python_path(codebase_path, codebase_modules)

def add_plot_path(plot_path):

    print "Some stuff"

def main():

    paths = {}

    with open('config.txt') as f:

        pairs = [line.split(': ')
                 for line in f]
        paths = {name : path[:-1]
                 for first, second in pairs}

    init_python_path(paths['codebase'])
    add_plot_path(paths['plots'])

if __name__=='__main__':
    main()
