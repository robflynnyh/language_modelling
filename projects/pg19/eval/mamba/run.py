import lming
from lming.loading.datasets.PG19Dataset import PG19TestDataset
import argparse


def main(args):
    # load dataset
    test_data = PG19TestDataset()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)