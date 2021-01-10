from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='penne',
    description='Pitch Estimating Neural NEtworks (PENNE)',
    version='0.0.1',
    author='Caedon Hsieh and Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/interactiveaudiolab/penne',
    install_requires=['librosa', 'resampy', 'scipy', 'torch', 'tqdm'],
    packages=['penne'],
    package_data={'penne': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['pitch', 'audio', 'speech', 'music', 'pytorch', 'crepe'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')