import os
from glob import glob
from setuptools import find_packages, setup, find_namespace_packages

package_name = 'uni_navid'

setup(
    name=package_name,
    version='0.0.0',
    # third_party.Uni-NaVid (hyphen) is auto-skipped by find_packages and is
    # installed separately via `pip install -e .` in the Dockerfile.
    packages=find_namespace_packages(
        include=[
            'uni_navid',
            'uni_navid.*',
            'third_party',
            'third_party.*',
            ],
    ),
    include_package_data = True,
    package_data={
        'third_party.UniNaVid.uninavid.processor': ['**/*.json'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='giorgio-marmolino',
    maintainer_email='giorgio.marmolino@gmail.com',
    description='Uni-NaVid VLA navigation stack for ROS 2 Humble',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'uninavid_node = uni_navid.navid_node:main',
            'action_node = uni_navid.action_node:main',
            'safety_layer_node = uni_navid.safety_layer_node:main',
            'instruction_node = uni_navid.instruction_node:main',
        ],
    },
)