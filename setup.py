from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='kanbansim',
      version='0.1',
      description='Agent-based Kanban process simulator',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='kanban simulator agent based',
      url='http://github.com/andrewchch/kanban_sim',
      author='Andrew Groom',
      author_email='andrewjgroom@gmail.com',
      license='MIT',
      packages=['kanbansim'],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['kanbansim=kanbansim.run:main'],
      },
      include_package_data=True,
      zip_safe=False)
