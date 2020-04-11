import subprocess

subprocess.call(
    "python -m pytest --ignore=data/ --verbosity=1", shell=True)
