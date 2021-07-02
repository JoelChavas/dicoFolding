from setuptools import setup, find_packages

setup(
    name='dicoFolding',
    version='0.0.1',
    packages=find_packages(exclude=['tests*', 'notebooks*']),
    license='CeCILL license version 2',
    description='Deep learning models to look for sulcus dictionary',
    long_description=open('README.rst').read(),
    install_requires=['deep_folding', 'pandas', 
                    'scipy', 'matplotlib',
                    'torch', 'tqdm',
                    'torchvision', 'torch-summary', 'hydra', 'hydra.core',
                    'dataclasses', 'hydra', 'OmegaConf',
                    'sklearn', 'scikit-image',
                    'pytorch-lightning', 'lightly',
                    'toolz', ],
    url='https://github.com/JoelChavas/dicoFolding',
    author='Joël Chavas',
    author_email='joel.chavas@cea.fr'
)
