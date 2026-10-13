import argparse
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.iotools import load_train_configs
from yaml import YAMLError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RDE evaluation")
    parser.add_argument("--config_file", required=True, type=Path,
                        help="YAML configuration saved with the trained model")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="RDE checkpoint to evaluate, such as best.pth or last.pth")
    parser.add_argument("--root_dir", type=Path,
                        help="override the dataset root directory saved in the YAML configuration")
    cli_args = parser.parse_args()

    config_file = cli_args.config_file.expanduser().resolve()
    checkpoint_file = cli_args.checkpoint.expanduser().resolve()
    if not config_file.is_file():
        parser.error(f"configuration file does not exist or is not a file: {config_file}")
    if not checkpoint_file.is_file():
        parser.error(f"checkpoint does not exist or is not a file: {checkpoint_file}")

    try:
        args = load_train_configs(str(config_file))
    except (OSError, YAMLError) as exc:
        parser.error(f"cannot load configuration file {config_file}: {exc}")
    if not args:
        parser.error(f"configuration file is empty: {config_file}")

    if cli_args.root_dir is not None:
        root_dir = cli_args.root_dir.expanduser().resolve()
        if not root_dir.is_dir():
            parser.error(f"dataset root directory does not exist or is not a directory: {root_dir}")
        args.root_dir = str(root_dir)

    args.training = False
    args.output_dir = str(checkpoint_file.parent)
    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    logger.info(f"Evaluating checkpoint: {checkpoint_file}")

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=str(checkpoint_file))
    model = model.cuda()
    do_inference(model, test_img_loader, test_txt_loader)


