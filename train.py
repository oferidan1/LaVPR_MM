import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule, IMAGENET_MEAN_STD
from sentence_transformers import SentenceTransformer
import os
import argparse
from model.LaVPR import LaVPR


def precompute_tau_from_data(model, datamodule, device, csv_dir):
    """One pass over training data to compute per-modality tau from similarity std.

    If CSV files from a prior run already exist in *csv_dir*, they are loaded
    instead of re-running inference (saves time across repeated experiments).

    Returns (tau_v, tau_l) — one float per modality.
    """
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    img_csv_path = csv_dir / 'vision_similarities.csv'
    text_csv_path = csv_dir / 'language_similarities.csv'

    if img_csv_path.exists() and text_csv_path.exists():
        print(f"[precomputed_tau] Loading cached similarities from {csv_dir}")
        img_sims = np.loadtxt(img_csv_path, delimiter=',', skiprows=1)
        text_sims = np.loadtxt(text_csv_path, delimiter=',', skiprows=1)
        tau_v = max(float(np.std(img_sims)), 0.01)
        tau_l = max(float(np.std(text_sims)), 0.01)
        print(f"  Vision  — mean={np.mean(img_sims):.6f}  std={np.std(img_sims):.6f}  → tau_v={tau_v:.6f}")
        print(f"  Language — mean={np.mean(text_sims):.6f}  std={np.std(text_sims):.6f}  → tau_l={tau_l:.6f}")
        return tau_v, tau_l

    print("[precomputed_tau] Computing per-modality similarity distributions over training data …")
    model.eval()

    train_loader = datamodule.train_dataloader()

    all_img_sims = []
    all_text_sims = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Precomputing similarities"):
            places, labels, texts = batch
            BS, N, ch, h, w = places.shape
            images = places.view(BS * N, ch, h, w).to(device)

            flat_texts = []
            for i in range(BS):
                for j in range(N):
                    flat_texts.append(texts[j][i])

            descriptors, text_embeds, _, _, _ = model(images, flat_texts)

            img_sim = torch.matmul(descriptors, descriptors.T)
            text_sim = torch.matmul(text_embeds, text_embeds.T)

            triu_mask = torch.triu(torch.ones(img_sim.shape[0], img_sim.shape[1],
                                              dtype=torch.bool, device=img_sim.device), diagonal=1)
            all_img_sims.append(img_sim[triu_mask].cpu().float().numpy())
            all_text_sims.append(text_sim[triu_mask].cpu().float().numpy())

    all_img_sims = np.concatenate(all_img_sims)
    all_text_sims = np.concatenate(all_text_sims)

    np.savetxt(img_csv_path, all_img_sims, delimiter=',', header='similarity', comments='')
    np.savetxt(text_csv_path, all_text_sims, delimiter=',', header='similarity', comments='')
    print(f"[precomputed_tau] Saved {len(all_img_sims)} pairwise similarities per modality to {csv_dir}")

    tau_v = max(float(np.std(all_img_sims)), 0.01)
    tau_l = max(float(np.std(all_text_sims)), 0.01)
    print(f"  Vision  — mean={np.mean(all_img_sims):.6f}  std={np.std(all_img_sims):.6f}  n={len(all_img_sims)}  → tau_v={tau_v:.6f}")
    print(f"  Language — mean={np.mean(all_text_sims):.6f}  std={np.std(all_text_sims):.6f}  n={len(all_text_sims)}  → tau_l={tau_l:.6f}")

    model.train()
    return tau_v, tau_l


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Resume parameters
    parser.add_argument("--vpr_dim", type=int, default=512, help="dimension of the vpr embeddings")
    parser.add_argument("--vpr_model_name", type=str, default="mixvpr")
    parser.add_argument("--vpr_model_backbone", type=str, default="ResNet50")
    # Other parameters
    parser.add_argument("--gpu", type=str, default='0', help="gpu id(s) to use")
    parser.add_argument("--device", type=str, default='cuda', help="device to train on (e.g. cuda, cuda:0, cpu)")    
    parser.add_argument("--epochs", type=int, default='10', help="number of epochs to train")    
    parser.add_argument("--train_csv", type=str, default="datasets/descriptions/gsv_cities_descriptions.csv")    
    parser.add_argument("--image_root", type=str, default="/mnt/d/data/gsv_cities/", help="root directory for images")
    #parser.add_argument("--val_csv", type=str, default="datasets/descriptions/pitts30k_val_descriptions.csv")    
    parser.add_argument("--val_csv", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")    
    parser.add_argument("--val_image_root", type=str, default="/mnt/d/data/pitts30k/images/val", help="root directory for images")
    parser.add_argument("--text_model_name", type=str, default="BAAI/bge-large-en-v1.5", help="text encoder model name")
    parser.add_argument("--is_freeze_text", type=int, default="1", help="freeze text encoder or not")
    parser.add_argument("--is_freeze_vpr", type=int, default="1", help="freeze vpr encoder or not")    
    parser.add_argument("--image_size", type=int, default="320", help="image size to vpr")
    parser.add_argument("--embeds_dim", type=int, default=512, help="dimension of the embeddings")
    parser.add_argument("--fusion_type", type=str, default='dynamic_weighting', help="type of fusion to use: mlp, add, dynamic_weighting, fixed_weighting")
    parser.add_argument("--is_encode_image", type=int, default="1", help="encode image or not")
    parser.add_argument("--is_encode_text", type=int, default="1", help="encode text or not")
    parser.add_argument("--is_text_pooling", type=int, default="1", help="pool text or not")
    parser.add_argument("--is_image_pooling", type=int, default="0", help="pool image or not")
    parser.add_argument("--is_pca", type=int, default="0", help="pool image or not")    
    parser.add_argument("--batch_size", type=int, default="120", help="batch size for training")
    parser.add_argument("--loss_name", type=str, default="MultiSimilarityLoss_Sij", help="name of the loss function to use")
    #parser.add_argument("--loss_name", type=str, default="MultiSimilarityLoss", help="name of the loss function to use")
    parser.add_argument("--is_val", type=int, default="1", help="run validation 0=no/1=yes")    
    parser.add_argument("--text_dim", type=int, default=1024, help="dimension of the text embeddings")
    parser.add_argument("--img_per_place", type=int, default=4, help="number of images per place")
    parser.add_argument("--use_dri", type=int, default=0, help="enable Differentiable Rank Integration (0=off, 1=on)")
    parser.add_argument("--dri_tau", type=float, default=None, help="fixed temperature for soft-rank sigmoid in DRI (mutually exclusive with --dri_dynamic_tau)")
    parser.add_argument("--dri_k", type=float, default=60.0, help="smoothing constant for reciprocal-rank fusion in DRI")
    parser.add_argument("--dri_dynamic_tau", type=int, default=0, help="use per-batch dynamic tau instead of fixed (0=fixed, 1=dynamic)")
    parser.add_argument("--dri_precomputed_tau", type=int, default=0, help="compute tau per-modality from train-data similarity std before training (0=off, 1=on)")
    parser.add_argument("--dri_sim_csv_dir", type=str, default=None, help="directory to save/load precomputed similarity CSVs (default: ./LOGS/sim_cache)")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name to log to (disabled if not set)")
    args = parser.parse_args()

    if args.use_dri:
        n_tau_strategies = sum([bool(args.dri_dynamic_tau), bool(args.dri_precomputed_tau), args.dri_tau is not None])
        if n_tau_strategies > 1:
            parser.error("--dri_tau, --dri_dynamic_tau, and --dri_precomputed_tau are mutually exclusive.")
        if n_tau_strategies == 0:
            parser.error("One of --dri_tau, --dri_dynamic_tau=1, or --dri_precomputed_tau=1 is required.")

    if args.dri_tau is None:
        args.dri_tau = 0.1

    return args            
            
if __name__ == '__main__':    
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    
    args = parse_arguments()
    if 'cuda' in args.device and ':' in args.device:
        gpu_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = 'cuda' if 'cuda' in args.device else args.device
    
    dataset_mean_std = IMAGENET_MEAN_STD
    image_size = args.image_size    
    
    val_set_names = []
    if args.is_val:
        val_set_names = ['pitts30k_val']    
        
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(image_size, image_size),
        num_workers=4,#28,
        show_data_stats=True,
        mean_std=dataset_mean_std,
        #val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
        val_set_names=val_set_names,
        train_image_root=args.image_root,
        train_csv=args.train_csv,
        val_image_root=args.val_image_root,
        val_csv=args.val_csv,
    )

    if args.fusion_type == 'text_adapter':
        args.is_text_pooling = 1

    # examples of backbones
    # resnet18, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
    # efficientnet_b0, efficientnet_b1, efficientnet_b2
    # swinv2_base_window12to16_192to256_22kft1k
    model = LaVPR(
        #---- Encoder
        vpr_model_name=args.vpr_model_name.lower(),
        vpr_model_backbone=args.vpr_model_backbone,
        vpr_encoder_dim=args.vpr_dim,
        
        #---- Train hyperparameters
        lr=0.05, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
        optimizer='sgd', # sgd, adamw
        weight_decay=0.001, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=650,
        #milestones=[2],
        milestones=[2,4,6,8],
        lr_mult=0.3,

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=args.loss_name,
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
        text_model_name=args.text_model_name,
        embeds_dim=args.embeds_dim,
        is_freeze_vpr=args.is_freeze_vpr,
        is_freeze_text=args.is_freeze_text,
        fusion_type=args.fusion_type,
        is_encode_image=args.is_encode_image,
        is_encode_text=args.is_encode_text,        
        is_text_pooling=args.is_text_pooling,
        is_image_pooling=args.is_image_pooling,        
        text_encoder_dim=args.text_dim,
        use_dri=args.use_dri,
        dri_tau=args.dri_tau,
        dri_k=args.dri_k,
        dri_dynamic_tau=args.dri_dynamic_tau,
    )
    
    # if args.is_encode_image and  args.vpr_resume_model is not None:
    #     model_state_dict = torch.load(args.vpr_resume_model)
    #     model.vpr_encoder.load_state_dict(model_state_dict)
        
    model = model.to(args.device)

    if args.use_dri and args.dri_precomputed_tau:
        csv_dir = args.dri_sim_csv_dir or './LOGS/sim_cache'
        tau_v, tau_l = precompute_tau_from_data(model, datamodule, args.device, csv_dir)
        dri = model.loss_fn.dri
        dri.precomputed_tau_v = torch.tensor(tau_v)
        dri.precomputed_tau_l = torch.tensor(tau_l)
        args.precomputed_tau_v = tau_v
        args.precomputed_tau_l = tau_l

    if args.is_val:    
        # model params saving using Pytorch Lightning
        # we save the best 3 models accoring to Recall@1 on pittsburg val
        checkpoint_cb = ModelCheckpoint(
            monitor='pitts30k_val/R1',
            filename=f'{"resnet50"}' +
            '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=3,
            mode='max',)
    else:
        checkpoint_cb = ModelCheckpoint(        
            filename=f'{"resnet50"}' +
            '_epoch({epoch:02d})_step({step:04d})',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=-1,
            every_n_epochs=1,
            mode='max',)

    #------------------
    # wandb logger (optional)
    wandb_logger = None
    if args.wandb_project:
        import wandb
        wandb.login(key="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")
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            config=vars(args),
        )

    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu' if 'cuda' in args.device else args.device,
        devices=[0] if 'cuda' in args.device else 'auto',
        default_root_dir=f'./LOGS/{"resnet50"}', # Tensorflow can be used to viz

        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=16, # we use half precision to reduce  memory usage
        max_epochs=args.epochs,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        logger=wandb_logger if wandb_logger else True,
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )
    
    # # Manually call validation
    #trainer.validate(model=model, datamodule=datamodule)

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
