import torch
import torch.nn.functional as F
import logging
import datetime
import os
from dataclasses import dataclass, asdict
import json

from models.mynet import MyNet
from utils.dataloader import get_test_dataloader, get_train_dataloader
from utils.tools import clip_gradient, save_model_summary
from test import run_test


@dataclass
class ExperimentConfig:
    # Experiment identification
    exp_name: str = "EMCAM_multi_feedback_CAMO"
    exp_description: str = "不用梯度检查点，迭代3次，每层都反馈，用CAMO测试"

    # Device configuration
    device: str = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Training batch parameters
    batch_size: int = 8  # Effective batch size will be batch_size * accumulation_steps
    accumulation_steps: int = 2  # Number of gradient accumulation steps

    # Optimization parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_value: float = 0.5

    # Scheduler parameters
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1

    # Training duration
    epochs: int = 100

    # Dataset parameters
    train_augmentation: bool = True
    test_dataset_path: str = "./Dataset/TestDataset/CAMO"

    # Auto test configuration
    auto_test: bool = True  # 是否在训练完成后自动运行测试

    def to_dict(self):
        return asdict(self)


def backup_code(exp_dir: str):
    """
    Backup important code files to the experiment directory
    """
    import shutil

    # Create a code backup directory
    backup_dir = os.path.join(exp_dir, "code_backup")
    os.makedirs(backup_dir, exist_ok=True)

    # List of files to backup
    files_to_backup = [
        "train.py",
        "test.py",
        "models/mynet.py"
    ]

    # Backup each file
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            # Preserve directory structure
            backup_path = os.path.join(backup_dir, file_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(file_path, backup_path)
        else:
            logging.warning(f"Could not backup {file_path}: File not found")


def setup_logger(config: ExperimentConfig, log_dir="logs"):
    # Create experiment-specific directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(log_dir, f"{config.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)

    # Backup code files
    backup_code(exp_dir)

    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create and set up file handler with experiment name
    file_handler = logging.FileHandler(
        os.path.join(exp_dir, f"training_log.txt")
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create and set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, exp_dir


def log_experiment_info(logger, config: ExperimentConfig):
    logger.info("=" * 50)
    logger.info(f"Experiment: {config.exp_name}")
    logger.info(f"Description: {config.exp_description}")
    logger.info("=" * 50)
    logger.info("Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)
    logger.info("=" * 50)


def train_loop(dataloader, model, optimizer, config: ExperimentConfig, logger):
    size = len(dataloader.dataset)
    model.train()
    for batch, item in enumerate(dataloader):
        imgs, gts = item["img"].to(config.device), item["gt"].to(config.device)

        # Compute prediction and loss
        P1, P2 = model(imgs)
        losses_p1 = [structure_loss(out, gts) for out in P1]
        loss_p1 = 0
        gamma = 0.2
        for it in range(len(P1)):
            loss_p1 += (gamma * it) * losses_p1[it]

        losses_p2 = [structure_loss(out, gts) for out in P2]
        loss_p2 = 0
        gamma = 0.2
        for it in range(len(P2)):
            loss_p2 += (gamma * it) * losses_p2[it]

        loss = loss_p1 + loss_p2
        loss.backward()  # Accumulate gradients without stepping

        if (batch + 1) % config.accumulation_steps == 0:
            clip_gradient(optimizer, config.gradient_clip_value)
            optimizer.step()
            optimizer.zero_grad()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * config.batch_size + len(imgs)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, logger, exp_dir):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    dataset_name = os.path.basename(config.test_dataset_path)

    with torch.no_grad():
        for item in dataloader:
            imgs, gts = item["img"].to(config.device), item["gt"].to(config.device)
            P1, P2 = model(imgs)
            test_loss += mae_loss(P1[-1] + P2[-1], gts).item() * imgs.shape[0]

    test_loss /= size
    global best_mae
    if test_loss < best_mae:
        best_mae = test_loss
        model_path = os.path.join(exp_dir, f"{config.exp_name}_best_model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"New best model saved with MAE: {best_mae:>8f} on {dataset_name}")
    logger.info(f"Test result on {dataset_name}: \n Avg MAE loss: {test_loss:>8f} Best MAE: {best_mae:>8f}\n")


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def mae_loss(pred: torch.Tensor, gt: torch.Tensor):
    pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    pred = pred.sigmoid()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = gt / (gt.max() + 1e-8)
    return F.l1_loss(pred, gt)


# Initialize configuration
config = ExperimentConfig()
best_mae = 1

try:
    # Setup logger and experiment directory
    logger, exp_dir = setup_logger(config)

    # Log experiment information
    log_experiment_info(logger, config)

    # Setup data
    train_dataloader = get_train_dataloader(
        batch_size=config.batch_size,
        use_augmentation=config.train_augmentation
    )
    test_dataloader = get_test_dataloader(
        test_data_root=config.test_dataset_path
    )

    # Setup model and optimization
    model = MyNet().to(config.device)

    # 保存模型统计信息
    save_model_summary(model, exp_dir, config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma
    )

    logger.info("Starting training...")
    for t in range(config.epochs):
        logger.info(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, config, logger)
        scheduler.step()
        test_loop(test_dataloader, model, logger, exp_dir)
    logger.info("Training completed successfully!")

    # 删除训练过程中占用的显存变量
    logger.info("Releasing GPU memory used during training...")
    # 删除模型和优化器
    del model, optimizer, scheduler
    # 删除数据加载器
    del train_dataloader, test_dataloader
    # 清空 GPU 缓存
    if "cuda" in config.device:
        torch.cuda.empty_cache()

    # 如果启用了自动测试，运行测试脚本
    if config.auto_test:
        logger.info("Starting automatic testing...")
        try:
            test_results = run_test(exp_dir, config.device)

            # 创建一个新的有序字典，确保 experiment_info 在最前面
            results = {
                'experiment_info': {
                    'name': config.exp_name,
                    'description': config.exp_description,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            # 添加测试结果
            results.update(test_results)

            # 保存测试结果
            result_path = os.path.join(exp_dir, "result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Test results saved to {result_path}")

            # 打印详细指标表格
            logger.info("\nTest Results:")
            logger.info("-" * 71)
            logger.info(f"{'Dataset':12} {'MAE↓':>10} {'Smeasure↑':>12} {'meanEm↑':>12} {'wFmeasure↑':>12}")
            logger.info("-" * 71)

            # 打印每个数据集的指标
            for result in results['per_dataset']:
                logger.info(f"{result['dataset']:12} "
                            f"{result['MAE']:10.3f} "
                            f"{result['Smeasure']:12.3f} "
                            f"{result['meanEm']:12.3f} "
                            f"{result['wFmeasure']:12.3f}")

            # 打印分隔线
            logger.info("-" * 71)

            # 打印平均指标
            avg = results['average']
            logger.info(f"{'Average':12} "
                        f"{avg['MAE']:10.3f} "
                        f"{avg['Smeasure']:12.3f} "
                        f"{avg['meanEm']:12.3f} "
                        f"{avg['wFmeasure']:12.3f}")
            logger.info("-" * 71)

            logger.info("\nAutomatic testing completed successfully!")
        except Exception as e:
            logger.error(f"Automatic testing failed with error: {str(e)}", exc_info=True)

except Exception as e:
    logger.error(f"Training failed with error: {str(e)}", exc_info=True)

finally:
    # Remove handlers to avoid adding them again if we run the script multiple times
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
