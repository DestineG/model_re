import os
import time
import datetime
import json
import logging
import torch
import torch.nn as nn
import wandb

from .data.dataset import get_data_loaders
from .model import ViT


def setup_logger(log_path):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def train_one_epoch(model, train_loader, criterion, optimizer, device, logger=None, epoch=None):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if logger is not None and i % 10 == 0:
            logger.info(f"Epoch: {epoch + 1} | Batch: {i + 1} | Loss: {loss.item():.4f} | Acc: {predicted.eq(labels).sum().item() / batch_size:.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def save_model_weights(model, save_path):
    torch.save(model.state_dict(), save_path)


def save_checkpoint(
        epoch,
        model,
        optimizer,
        scheduler,
        best_acc,
        save_path
    ):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
    }

    torch.save(checkpoint, save_path)


def main():
    model_name = "ViT"
    experiment_name = "vit_base_run1"

    epochs = 100
    batch_size = 128
    lr = 1e-4
    save_interval = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = os.path.join("./checkpoints", model_name, experiment_name)
    weights_dir = os.path.join(experiment_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    log_path = os.path.join(experiment_dir, "train.log")
    logger = setup_logger(log_path)

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "model": model_name,
        "device": str(device),
        "save_interval": save_interval,
        "experiment_dir": experiment_dir,
        "weights_dir": weights_dir,
    }

    logger.info("=" * 80)
    logger.info("Experiment Config")
    logger.info(json.dumps(config, indent=4, ensure_ascii=False))
    logger.info("=" * 80)

    run = wandb.init(
        project=f"cv-research-project-{model_name.lower()}",
        name=experiment_name,
        config=config
    )

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    if model_name == "ViT":
        model = ViT(img_size=(32, 32), patch_size=4, num_classes=10).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    best_acc = 0.0

    logger.info("Start training")

    for epoch in range(epochs):
        current_epoch = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger,
            epoch=epoch
        )

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        run.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "train/lr": current_lr,
        }, step=current_epoch)

        # 保存最新模型权重
        save_model_weights(
            model,
            os.path.join(weights_dir, "model_latest.pth")
        )

        # 每隔 save_interval 轮保存一次模型权重
        if current_epoch % save_interval == 0:
            epoch_weight_path = os.path.join(
                weights_dir,
                f"model_epoch_{current_epoch}.pth"
            )
            save_model_weights(model, epoch_weight_path)
            logger.info(f"Saved epoch weight: {epoch_weight_path}")

        # 保存最佳模型权重
        if test_acc > best_acc:
            best_acc = test_acc

            run.summary["best/test_acc"] = best_acc
            run.summary["best/epoch"] = current_epoch

            best_weight_path = os.path.join(weights_dir, "model_best.pth")
            save_model_weights(model, best_weight_path)

            logger.info(
                f"Updated best model at epoch {current_epoch}: "
                f"best_acc={best_acc:.4f}, path={best_weight_path}"
            )

        # 更新学习率
        scheduler.step()

        # 保存完整训练状态，方便断点续训
        checkpoint_path = os.path.join(weights_dir, "checkpoint_latest.pth")
        save_checkpoint(
            epoch=current_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_acc=best_acc,
            save_path=checkpoint_path
        )

        remaining_seconds = (time.time() - start_time) / (current_epoch + 1) * (epochs - current_epoch)
        eta_str = str(datetime.timedelta(seconds=int(remaining_seconds)))
        logger.info(
            f"Epoch [{current_epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.4f} "
            f"Best Acc: {best_acc:.4f} "
            f"LR: {current_lr:.6f} | "
            f"ETA: {eta_str}"
        )

    logger.info(f"Best Test Acc: {best_acc:.4f}")

    run.finish()

if __name__ == "__main__":
    main()