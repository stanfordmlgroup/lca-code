"""Pre-process DICOMs and create an HDF5 file."""
import argparse


def main(args):
    raise NotImplementedError('Pre-process not implemented.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process DICOMs')

    main(parser.parse_args())
