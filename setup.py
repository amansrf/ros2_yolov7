from setuptools import setup

package_name = 'ros_yolov7'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'models', 'utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roar',
    maintainer_email='roar@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = ros_yolov7.inference_node:main'
        ],
    },
)
