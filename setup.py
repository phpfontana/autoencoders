from setuptools import setup, find_packages

setup(
    name='autoencoders',
    version='0.0.1',
    author='P. Fontana',
    description='Autoencoders models',
    url='https://github.com/phpfontana/autoencoders/tree/main',
    packages=find_packages(where='autoencoders'),  # Finds packages inside 'autoencoders' directory
    package_dir={'': 'autoencoders'},  # Points to the directory where packages are located
    install_requires=[
        'torch==2.4.0',
        'torchaudio==2.4.0',
        'torchvision==0.19.0'
    ],
    python_requires='>=3.6',  # Specify minimum Python version
)