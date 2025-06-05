#!/usr/bin/env python3
"""MMVC Training CLI."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# プロジェクトルートをパスに追加
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.core.data.dataset import TextAudioSpeakerDataset, TextAudioSpeakerCollate
from src.core.training.losses import (
    generator_loss, discriminator_loss, feature_loss, kl_loss
)
from src.core.models.vits import SynthesizerTrn
from src.core.models.discriminators import MultiPeriodDiscriminator
from src.utils.common import (
    setup_logging, save_checkpoint, load_checkpoint, 
    get_latest_checkpoint
)
from src.utils.audio import mel_spectrogram


def setup_distributed(rank: int, world_size: int, port: str = "12355"):
    """分散学習の初期化."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # NCCLバックエンドを使用（Linux/GPU）
    # Windowsの場合はgloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def cleanup_distributed():
    """分散学習のクリーンアップ."""
    dist.destroy_process_group()


def train_epoch(rank, epoch, config, models, optimizers, schedulers, scaler, loaders, writer):
    """1エポックの学習を実行する."""
    net_g, net_d = models
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    
    net_g.train()
    net_d.train()
    
    for batch_idx, batch in enumerate(train_loader):
        if torch.cuda.is_available():
            batch = [x.cuda(rank, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch]
        
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch
        
        # Generator forward pass
        with torch.cuda.amp.autocast(enabled=config["train"]["fp16_run"]):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)
            
            mel = mel_spectrogram(
                y.squeeze(1), 
                config["data"]["filter_length"],
                config["data"]["n_mel_channels"],
                config["data"]["sampling_rate"],
                config["data"]["hop_length"],
                config["data"]["win_length"],
                config["data"]["mel_fmin"],
                config["data"]["mel_fmax"]
            )
            
            y_mel = mel_spectrogram(
                y_hat.squeeze(1),
                config["data"]["filter_length"],
                config["data"]["n_mel_channels"],
                config["data"]["sampling_rate"],
                config["data"]["hop_length"],
                config["data"]["win_length"],
                config["data"]["mel_fmin"],
                config["data"]["mel_fmax"]
            )
            
            y = y.unsqueeze(1) if y.dim() == 2 else y
            y_hat = y_hat.unsqueeze(1) if y_hat.dim() == 2 else y_hat
            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
        
        # Update discriminator
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), config["train"]["grad_clip"])
        scaler.step(optim_d)
        
        with torch.cuda.amp.autocast(enabled=config["train"]["fp16_run"]):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            
            loss_mel = F.l1_loss(mel, y_mel) * config["train"]["c_mel"]
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + l_length
        
        # Update generator
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), config["train"]["grad_clip"])
        scaler.step(optim_g)
        scaler.update()
        
        # Logging
        if batch_idx % config["train"]["log_interval"] == 0 and rank == 0:
            global_step = epoch * len(train_loader) + batch_idx
            lr = scheduler_g.get_last_lr()[0]
            
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss G: {loss_gen_all.item():.6f}\t'
                  f'Loss D: {loss_disc_all.item():.6f}\t'
                  f'LR: {lr:.6f}')
            
            if writer is not None:
                writer.add_scalar('train/loss_g', loss_gen_all.item(), global_step)
                writer.add_scalar('train/loss_d', loss_disc_all.item(), global_step)
                writer.add_scalar('train/loss_mel', loss_mel.item(), global_step)
                writer.add_scalar('train/loss_kl', loss_kl.item(), global_step)
                writer.add_scalar('train/loss_fm', loss_fm.item(), global_step)
                writer.add_scalar('train/learning_rate', lr, global_step)
                writer.add_scalar('train/grad_norm_g', grad_norm_g, global_step)
                writer.add_scalar('train/grad_norm_d', grad_norm_d, global_step)
        
        # Save checkpoint
        if batch_idx % config["train"]["checkpoint_interval"] == 0 and rank == 0:
            global_step = epoch * len(train_loader) + batch_idx
            save_checkpoint(
                net_g,
                optim_g,
                scheduler_g.get_last_lr()[0], 
                global_step,
                os.path.join(config["model_dir"], f"G_{global_step:06d}.pth")
            )
            save_checkpoint(
                net_d,
                optim_d,
                scheduler_d.get_last_lr()[0], 
                global_step,
                os.path.join(config["model_dir"], f"D_{global_step:06d}.pth")
            )
        
        # 学習率スケジューリング
        scheduler_g.step()
        scheduler_d.step()


def train_worker(rank: int, world_size: int, config):
    """ワーカープロセスの学習ループ."""
    # 分散学習の初期化
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # ロガーの設定
    setup_logging(verbose=True)
    
    # GPUデバイスの設定
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    
    # データセットとデータローダー
    train_dataset = TextAudioSpeakerDataset(config["data"]["training_files"], config)
    eval_dataset = TextAudioSpeakerDataset(config["data"]["validation_files"], config)
    
    collate_fn = TextAudioSpeakerCollate()
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        eval_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["train"]["batch_size"],
        sampler=eval_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    # モデルの初期化
    net_g = SynthesizerTrn(
        n_vocab=config["data"]["n_vocab"],
        spec_channels=config["data"]["filter_length"] // 2 + 1,
        segment_size=config["train"]["segment_size"] // config["data"]["hop_length"],
        **config["model"]
    ).to(device)
    
    net_d = MultiPeriodDiscriminator(config["model"].get("use_spectral_norm", False)).to(device)
    
    # 分散学習用にラップ
    if world_size > 1:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    
    # オプティマイザーとスケジューラー
    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        config["train"]["learning_rate"], 
        betas=config["train"]["betas"], 
        eps=config["train"]["eps"]
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), 
        config["train"]["learning_rate"], 
        betas=config["train"]["betas"], 
        eps=config["train"]["eps"]
    )
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config["train"]["lr_decay"])
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config["train"]["lr_decay"])
    
    # 混合精度学習
    scaler = torch.cuda.amp.GradScaler(enabled=config["train"]["fp16_run"])
    
    # TensorBoardライター（ランク0のみ）
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(config["model_dir"], "logs"))
    else:
        writer = None
    
    # 学習ループ
    for epoch in range(config["train"]["epochs"]):
        if rank == 0:
            print(f"Starting epoch {epoch}")
        
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # 学習関数を呼び出し
        train_epoch(
            rank, epoch, config, 
            (net_g, net_d), 
            (optim_g, optim_d), 
            (scheduler_g, scheduler_d), 
            scaler, 
            (train_loader, eval_loader), 
            writer
        )
    
    # TensorBoardライターを閉じる
    if rank == 0 and writer is not None:
        writer.close()
    
    # 分散学習のクリーンアップ
    if world_size > 1:
        cleanup_distributed()


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description='MMVC Trainer')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='設定ファイルのパス')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='モデル保存ディレクトリ')
    parser.add_argument('--resume', type=str, default='',
                       help='学習再開用のチェックポイント')
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # モデル保存ディレクトリの設定
    config["model_dir"] = args.model
    os.makedirs(config["model_dir"], exist_ok=True)
    
    # 設定をコピー
    import shutil
    shutil.copy(args.config, os.path.join(config["model_dir"], "config.json"))
    
    # GPUの数を取得
    n_gpus = torch.cuda.device_count()
    
    if n_gpus > 1:
        # マルチGPU学習
        mp.spawn(train_worker, nprocs=n_gpus, args=(n_gpus, config))
    else:
        # シングルGPU学習
        train_worker(0, 1, config)


if __name__ == "__main__":
    main()


def train_epoch(rank: int, epoch: int, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    """1エポックの学習を実行する."""
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    writer, writer_eval = writers
    
    train_loader.sampler.set_epoch(epoch)
    
    net_g.train()
    net_d.train()
    
    global_step = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # データをGPUに移動
        if torch.cuda.is_available():
            batch = [x.cuda(rank, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch]
        
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch
        
        # 生成器の学習
        optim_g.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=hps.train.fp16_run):
            # TODO: モデルのフォワードパスを実装
            pass
        
        # 識別器の学習
        optim_d.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=hps.train.fp16_run):
            # TODO: 識別器の学習を実装
            pass
        
        global_step += 1
        
        # ログ出力
        if rank == 0 and global_step % hps.train.log_interval == 0:
            logger.info(f"Epoch: {epoch}, Step: {global_step}")
        
        # 評価とチェックポイント保存
        if rank == 0 and global_step % hps.train.eval_interval == 0:
            evaluate(net_g, eval_loader, writer_eval, global_step, hps)
            save_checkpoint(
                net_g, 
                optim_g, 
                scheduler_g.get_last_lr()[0],
                global_step,
                os.path.join(hps.model_dir, f"G_{global_step:06d}.pth")
            )
            save_checkpoint(
                net_d,
                optim_d,
                scheduler_d.get_last_lr()[0], 
                global_step,
                os.path.join(hps.model_dir, f"D_{global_step:06d}.pth")
            )
        
        # 学習率スケジューリング
        scheduler_g.step()
        scheduler_d.step()


def evaluate(net_g, eval_loader, writer_eval, global_step, hps):
    """評価を実行する."""
    net_g.eval()
    
    eval_loss = 0.0
    eval_steps = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if torch.cuda.is_available():
                batch = [x.cuda(0, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch]
            
            # TODO: 評価ロジックを実装
            eval_steps += 1
            
            if batch_idx >= 10:  # 評価は10バッチまで
                break
    
    avg_eval_loss = eval_loss / eval_steps if eval_steps > 0 else 0.0
    
    if writer_eval is not None:
        writer_eval.add_scalar('eval/loss', avg_eval_loss, global_step)
    
    net_g.train()


def train_worker(rank: int, world_size: int, hps):
    """ワーカープロセスの学習ループ."""
    # 分散学習の初期化
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # ロガーの設定
    logger = setup_logger(f"train_worker_{rank}")
    
    # ランダムシードの設定
    set_random_seed(hps.train.seed + rank)
    
    # GPUデバイスの設定
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    
    # データセットとデータローダー
    train_dataset = TextAudioSpeakerDataset(hps.data.training_files, hps)
    eval_dataset = TextAudioSpeakerDataset(hps.data.validation_files, hps, augmentation=False)
    
    collate_fn = TextAudioSpeakerCollate()
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        eval_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=hps.train.batch_size,
        sampler=eval_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    # TODO: モデルの初期化
    # net_g = Generator(hps).to(device)
    # net_d = Discriminator(hps).to(device)
    
    # 分散学習用にラップ
    # if world_size > 1:
    #     net_g = DDP(net_g, device_ids=[rank])
    #     net_d = DDP(net_d, device_ids=[rank])
    
    # オプティマイザーとスケジューラー
    # optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    # optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    
    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)
    
    # 混合精度学習
    scaler = torch.cuda.amp.GradScaler(enabled=hps.train.fp16_run)
    
    # TensorBoardライター（ランク0のみ）
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(hps.model_dir, "logs"))
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "logs", "eval"))
    else:
        writer = None
        writer_eval = None
    
    # 学習ループ
    for epoch in range(hps.train.epochs):
        logger.info(f"Starting epoch {epoch}")
        
        # TODO: 学習関数を呼び出し
        # train_epoch(
        #     rank, epoch, hps, 
        #     (net_g, net_d), 
        #     (optim_g, optim_d), 
        #     (scheduler_g, scheduler_d), 
        #     scaler, 
        #     (train_loader, eval_loader), 
        #     logger, 
        #     (writer, writer_eval)
        # )
    
    # TensorBoardライターを閉じる
    if rank == 0:
        writer.close()
        writer_eval.close()
    
    # 分散学習のクリーンアップ
    if world_size > 1:
        cleanup_distributed()


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description='MMVC Trainer')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='設定ファイルのパス')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='モデル保存ディレクトリ')
    parser.add_argument('-fg', '--fine_tuning_g', type=str, default='',
                       help='生成器のFine tuning用モデル')
    parser.add_argument('-fd', '--fine_tuning_d', type=str, default='',
                       help='識別器のFine tuning用モデル')
    parser.add_argument('--resume', type=str, default='',
                       help='学習再開用のチェックポイント')
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    hps = get_hparams_from_file(args.config)
    
    # モデル保存ディレクトリの設定
    hps.model_dir = args.model
    os.makedirs(hps.model_dir, exist_ok=True)
    
    # 設定をコピー
    import shutil
    shutil.copy(args.config, os.path.join(hps.model_dir, "config.json"))
    
    # GPUの数を取得
    n_gpus = torch.cuda.device_count()
    
    if n_gpus > 1:
        # マルチGPU学習
        mp.spawn(train_worker, nprocs=n_gpus, args=(n_gpus, hps))
    else:
        # シングルGPU学習
        train_worker(0, 1, hps)


if __name__ == "__main__":
    main()
