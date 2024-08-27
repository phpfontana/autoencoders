from setuptools import setup, find_packages

setup(
    name='autoencoders',
    version='0.0.1',
    author='Fontana, P.',
    description='Autoencoders models',
    url='https://github.com/phpfontana/autoencoders/tree/main',
    package_dir={'': 'autoencoders'},
    packages=find_packages(where='autoencoders'),
    install_requires=[
        'torch==2.4.0',
        'torchaudio==2.4.0',
        'torchvision==0.19.0'
    ],
)