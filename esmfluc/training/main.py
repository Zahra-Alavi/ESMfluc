"""Train one of the published ESMfluc conditions."""
import logging
import torch
from esmfluc.training.arguments import parse_arguments


def main():
    args = parse_arguments().parse_args()
    args.device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO)
    from esmfluc.training.train import train
    train(args)


if __name__ == "__main__":
    main()
