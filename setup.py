from distutils.core import setup

setup(
    name='UQGrid',
    version='0.1dev',
    packages=['uqgrid',],
    license='MIT',
    long_description=open('README.md').read(),

    author='D. Adrian Maldonado',
    author_email='maldonadod@anl.gov',

    test_suite='nose.collector',
    tests_require=['nose'],
)