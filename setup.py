import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="u2net",
    version="1.2.0",
    url="https://github.com/David-Estevez/U-2-Net/",
    author="David Estevez",
    description='Implementation of "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection." packaged for easy use.',
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(),
    install_requires=requirements,
)
