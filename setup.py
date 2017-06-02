from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='omtools',
      version='0.1',
      description='One- and two-dimensional objective mapping.',
      url='https://github.com/apaloczy/omtools',
      license='MIT',
      packages=['omtools'],
      install_requires=['numpy'],
      test_suite = 'nose.collector',
      zip_safe=False)
