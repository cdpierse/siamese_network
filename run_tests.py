import subprocess

subprocess.call(
    "python -m pytest --ignore=data/ --verbosity=1", shell=True)

# import torchvision
# dataset = torchvision.datasets.CelebA('data/',download= True)
