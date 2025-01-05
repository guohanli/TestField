from pathlib import Path
import os
import json
import sys
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import py_sod_metrics
from tqdm import tqdm

from utils.dataloader import get_test_dataloader
import importlib.util


# 从 exp_dir 动态导入 MyNet
def import_mynet_from_exp_dir(exp_dir):
    # 构建 mynet.py 的路径
    mynet_path = Path(exp_dir) / "code_backup" / "models" / "mynet.py"

    # 使用 importlib 动态加载模块
    spec = importlib.util.spec_from_file_location("mynet", mynet_path)
    mynet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mynet_module)

    # 返回 MyNet 类
    return mynet_module.MyNet


class Config:
    # 设备配置
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # 路径配置
    weight_path = None  # 将在运行时设置
    exp_dir = None  # 将在运行时设置
    test_dataset_path = Path("./Dataset/TestDataset")

    # 数据集配置
    datasets = ["CAMO", "CHAMELEON", "COD10K", "NC4K"]

    @classmethod
    def setup(cls, exp_dir, device):
        """设置实验目录和权重路径"""
        cls.exp_dir = Path(exp_dir)
        cls.weight_path = list(cls.exp_dir.glob("*best_model.pth"))[0]
        cls.result_imgs_path = cls.exp_dir / "result_imgs"
        if device:
            cls.device = device


def save_binary_image(image_path, binary_image_tensor):
    """将二值图像张量保存为PNG格式的图像文件。"""
    to_pil = transforms.ToPILImage()
    binary_image_pil = to_pil(binary_image_tensor.squeeze(0))
    binary_image_pil.save(image_path)


def test(dataloader, model, device, dataset_name):
    """执行模型测试"""
    model.eval()
    with torch.no_grad():
        desc = f"Generating Images for {dataset_name:10s}" if dataset_name else "Generating Images"
        for item in tqdm(dataloader, desc=desc, unit="img", file=sys.stdout, colour='green'):
            imgs, gts = item["img"].to(device), item["gt"].to(device)
            P1, P2 = model(imgs)
            pred = P1[-1] + P2[-1]
            pred = F.interpolate(pred, size=gts.shape[-2:], mode="bilinear", align_corners=False)
            pred = pred.sigmoid()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            gt_path = item["gt_path"][0]
            original_path = Path(gt_path)
            folder_name = original_path.parts[-3]
            file_name = original_path.parts[-1]

            target_folder = Config.result_imgs_path / folder_name
            target_path = target_folder / file_name

            target_folder.mkdir(parents=True, exist_ok=True)
            save_binary_image(target_path, pred)


def cal_metrics(mask_root, pred_root, dataset_name):
    """计算评估指标"""
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()

    mask_name_list = sorted(os.listdir(mask_root))
    desc = f"Processing {dataset_name:10s}" if dataset_name else "Processing"
    for mask_name in tqdm(mask_name_list, unit="img", desc=desc, file=sys.stdout, colour='green'):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    result = {
        "dataset": dataset_name,
        "MAE": mae,
        "Smeasure": sm,
        "meanEm": em["curve"].mean(),
        "wFmeasure": wfm,
    }

    print(f"\n{dataset_name:9s}  "
          f"MAE:{result['MAE']:.3f}  "
          f"Smeasure:{result['Smeasure']:.3f}  "
          f"meanEm:{result['meanEm']:.3f}  "
          f"wFmeasure:{result['wFmeasure']:.3f}")

    return result


def run_test(exp_dir, device=None):
    """运行测试流程"""
    # 设置配置
    Config.setup(exp_dir, device)
    print(f"Using {Config.device} device")

    # 动态导入 MyNet
    MyNet = import_mynet_from_exp_dir(exp_dir)

    # 加载模型
    model = MyNet().to(Config.device)
    model.load_state_dict(torch.load(Config.weight_path))

    # 测试阶段
    for dataset in Config.datasets:
        test_data_root = Config.test_dataset_path / dataset
        test_dataloader = get_test_dataloader(test_data_root)
        test(test_dataloader, model, Config.device, dataset)
    print("Generating Images Done!")

    # 评估阶段
    results = []
    for dataset in Config.datasets:
        pred_root = Config.result_imgs_path / dataset
        mask_root = Config.test_dataset_path / dataset / "GT"
        result = cal_metrics(
            mask_root=mask_root,
            pred_root=pred_root,
            dataset_name=dataset
        )
        results.append(result)

    print("\nSummary:")
    for result in results:
        print(f"{result['dataset']:9s}  "
              f"MAE:{result['MAE']:.3f}  "
              f"Smeasure:{result['Smeasure']:.3f}  "
              f"meanEm:{result['meanEm']:.3f}  "
              f"wFmeasure:{result['wFmeasure']:.3f}")

    # 创建最终结果字典
    final_results = {
        "per_dataset": results,
        "average": {
            "MAE": sum(r["MAE"] for r in results) / len(results),
            "Smeasure": sum(r["Smeasure"] for r in results) / len(results),
            "meanEm": sum(r["meanEm"] for r in results) / len(results),
            "wFmeasure": sum(r["wFmeasure"] for r in results) / len(results)
        }
    }

    return final_results


if __name__ == "__main__":
    exp_dir = "logs/EMCAM_multi_feedback_20250104_213001"
    results = run_test(exp_dir)

    # 保存测试结果
    result_path = os.path.join(exp_dir, "result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
