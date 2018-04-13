from setuptools import setup, find_packages

setup(name='gridMARL',
      version='0.0.1',
      author='David Earl Hostallero',
      author_email='ddhostallero@kaist.ac.kr',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym']
)
