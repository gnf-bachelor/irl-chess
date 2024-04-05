from setuptools import setup, find_packages

setup(
    name='project',
    version='0.1.0',
    author='Gabriel Agerholm Ruge',
    #author_email='your.email@example.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/yourproject',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    # classifiers=[
    #     'Development Status :: 3 - Alpha',
    #     'Intended Audience :: Developers',
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.7',
    #     'Programming Language :: Python :: 3.8',
    #     'Programming Language :: Python :: 3.9',
    # ],
)