import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('alias_copyi_CenterlessClustering', parent_package, top_path)
    config.add_subpackage('KSUMS2')
    config.add_subpackage('KSUMSX_eigen')
    config.add_subpackage('Public')
    config.add_data_dir('data')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
