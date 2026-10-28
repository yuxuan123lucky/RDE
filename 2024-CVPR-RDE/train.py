import os
import os.path as op
import torch
import numpy as np
import random
import time
from pathlib import Path


from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from model.local_evidence import validate_evidence_options
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    evidence_enabled = args.evidence_mode == 'a_only'
    if evidence_enabled:
        validate_evidence_options(args.evidence_candidates, args.evidence_weight,
                                  args.evidence_temperature, args.evidence_margin)
        if args.noisy_rate != 0.0 or 'TAL' not in args.loss_names.split('+'):
            raise ValueError('A-only requires --noisy_rate 0.0 and the original TAL task')
        if args.distributed:
            raise ValueError('the mechanism A pilot supports one GPU')
        outputs = (Path(__file__).resolve().parents[1] / 'outputs').resolve()
        root = outputs if args.output_dir == 'logs' else Path(args.output_dir).expanduser().resolve()
        if root != outputs and outputs not in root.parents:
            raise ValueError(f'A-only outputs must be under {outputs}')
        args.output_dir = str(root)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if evidence_enabled:
        name = name + '_evidence_a'
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}_{args.loss_names}')
    if evidence_enabled:
        # Atomically reserve a fresh run before opening logs/configs/checkpoints.
        if outputs not in Path(args.output_dir).resolve().parents:
            raise ValueError('A-only run path escaped the repository outputs directory')
        os.makedirs(args.output_dir, exist_ok=False)
    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    if not os.path.isdir(args.output_dir+'/img'):
        os.makedirs(args.output_dir+'/img')
    # get image-text pair datasets dataloader

    # if 'ICFG-PEDES' not in args.dataset_name: #fixed
    #     args.val_dataset = 'val'
        
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        if evidence_enabled:
            expected_evidence = {key: getattr(args, key) for key in (
                'evidence_mode', 'evidence_candidates', 'evidence_weight',
                'evidence_temperature', 'evidence_margin')}
            if checkpoint.get('evidence_config') != expected_evidence:
                raise ValueError('A-only resume requires matching evidence_config metadata; '
                                 'resume is not weight-only Baseline initialization')
        start_epoch = checkpoint['epoch']
        logger.info(f"===================>start {start_epoch}")


    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
    
    # Keep the original Test block; smoke runs can explicitly avoid Test access.
    if not args.skip_final_test:
        logger.info(f"===================>start test")
        args.training = False
        test_img_loader, test_txt_loader, num_classes = build_dataloader(args)

        asss = ['best.pth','last.pth']
        for i in range(len(asss)):
            if os.path.exists(op.join(args.output_dir, asss[i])):
                model = build_model(args,num_classes)
                checkpointer = Checkpointer(model)
                checkpointer.load(f=op.join(args.output_dir, asss[i]))
                model = model.cuda()
                do_inference(model, test_img_loader, test_txt_loader)
     
