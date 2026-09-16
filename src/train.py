import gc
import os, sys, argparse, time
import logging, datetime
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR

# from LossFunctions.MaskedLoss import MaskedMSE
# from LossFunctions.SSIM_Huber import LossSSIMHuber
# from LossFunctions.Weighted_MSE import Weighted_MSE
# from LossFunctions.FpTMSE import FpTMSELoss
# from LossFunctions.Charbonnier import CharbonnierLoss

from src.dataset_processing import MyDataset_3d
# from src.dataset_processing import MyDataset_2d
# from src.dataset_processing import MyDataset_5d
from utils import *
from torchmetrics.functional import structural_similarity_index_measure as SSIM

# from transformers import get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description='recon_training')
parser.add_argument("--batchSize", type=int, default=64,
                    help="Training batch size")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs")
parser.add_argument("--wnum", type=int, default=4,
                    help="Number of workers")
parser.add_argument("--gpunum", type=int, default=1,
                    help="Number of GPUs")
parser.add_argument("--milestone", type=int, default=15,
                    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Initial learning rate")
parser.add_argument("--net", type=str, default='MambaIR',
                    help="Selected Neural Network")
parser.add_argument("--dpath", type=str, default='./pre_data',
                    help="Path to the training dataset")
parser.add_argument("--thre", type=int, default=100,
                    help="Threshold to use GPU")
parser.add_argument("--dtype", type=str, default='3d',
                    help="2d: 2d data; 3d: 3d data; 5d: 5d data")
parser.add_argument("--loss", type=str, default='MSE',
                    help="Selected Loss Function")
parser.add_argument("--cp", type=str2bool, default=False,
                    help="Whether to use checkpoint to continue uncompleted training")
parser.add_argument("--mfolder", type=str, default='',
                    help="Folder to model; available when [--cp] is True")

opt = parser.parse_args()




current_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

os.makedirs(f'./trained_models/{opt.dtype}/', exist_ok=True)
save_dir = f'./trained_models/{opt.dtype}/{opt.net}{current_time}'

def main():
    # 建立模型
    # 导入网络的模块
    from importlib import import_module
    # match opt.dtype:
    #     case '2d':
    #         net_folder = "networks.2d."
    #     case '3d':
    #         net_folder = "networks._3d."
    #     case '5d':
    #         net_folder = "networks.5d."

    net_folder = "networks."
    module_name = net_folder + opt.net
    print(f"Import from: {module_name}")

    net_module = import_module(module_name)
    net_class = getattr(net_module, opt.net)
    net = net_class()
    print("Loading model: ", opt.net)
    # 采用多GPU进行训练
    device_ids = multi_gpu(opt.gpunum, opt.thre)
    cuda_visible_devices = ",".join(str(id) for id in device_ids)  # 将 device_ids 转换为逗号分隔的字符串
    # print(device_ids, cuda_visible_devices)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device(f'cuda:{device_ids[0]}')
    # 加载模型和损失函数，同时放入device中
    model = nn.DataParallel(net, device_ids=device_ids).to(device)

    match opt.loss:
        case 'MSE':
            criterion = nn.MSELoss().to(device)
        # case 'WMSE':
        #     criterion = Weighted_MSE(weight_type='exp', alpha=8).to(device)
        # case 'MaskedMSE':  # not useful
        #     criterion = MaskedMSE().to(device)
        # case 'SSIM':  # not useful
        #     criterion = SSIMLoss().to(device)
        case 'L1':
            criterion = nn.L1Loss().to(device)
        # case 'SSIM_Huber':
        #     criterion = LossSSIMHuber(delta=0.03).to(device)
        # case 'FpTMSE':
        #     criterion = FpTMSELoss().to(device)
        # case 'Charbonnier':
        #     criterion = CharbonnierLoss().to(device)

    # criterion = custom_loss().to(device)

    # 加载数据集
    print('Loading dataset ...\n')
    dpath = f'./dataset/{opt.dtype}/' + opt.dpath + '.h5'
    match opt.dtype:
        case '2d':
            dataset_train = MyDataset(datapath=dpath, istrain=True)
            dataset_val = MyDataset(datapath=dpath, istrain=False)
        case '3d':
            dataset_train = MyDataset_3d(dataset_fname=dpath, istrain=True)
            dataset_val = MyDataset_3d(dataset_fname=dpath, istrain=False)
        case '5d':
            dataset_train = MyDataset_5d(datapath=dpath, istrain=True)
            dataset_val = MyDataset_5d(datapath=dpath, istrain=False)


    train_loader = DataLoader(dataset=dataset_train, num_workers=opt.wnum, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    print("## Training batches: %d" % int(len(train_loader)))
    val_loader = DataLoader(dataset=dataset_val, num_workers=opt.wnum, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    print("## Validating batches: %d" % (int(len(val_loader)) if val_loader is not None else 0))

    total_iterations = len(train_loader) * opt.epochs
    print(f"Total iterations: {total_iterations}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_iterations,
        eta_min=1e-6
    )

    if opt.cp:
        # if opt.dtype == '2d':
        #     read_save_dir = f'./trained_models/{opt.mfolder}'
        # elif opt.dtype == '3d':
        #     read_save_dir = f'./trained_models_3d/{opt.mfolder}'
        read_save_dir = f'./trained_models/{opt.dtype}/{opt.mfolder}'
        ck_filepath = f'{read_save_dir}/checkpoint.pth'
        start_epoch, results_train, results_val, timelist = (
            load_checkpoint(model, optimizer, scheduler, ck_filepath))
        assert results_train is not None
        assert results_val is not None
        assert timelist is not None
    else:
        start_epoch = 0
        results_train = {'Loss':[], 'PSNR': [], 'SSIM': []}
        results_val = {'Loss':[], 'PSNR': [], 'SSIM': []}
        timelist = []





    # input('Press "Enter" to continue...')
    # 用于记录loss和SNR

    # 训练
    best = 0.
    try:
        for epoch in range(start_epoch, opt.epochs):
            # print(f"Allocated memory pos1: {torch.cuda.memory_allocated(device=device) / (1024 * 1024):.4f} MB")
            print(f'Epoch-{epoch+1} begin.')
            print('Learning rate %f' % optimizer.param_groups[0]['lr'])

            model.train()  # 训练模式
            loss_list = []
            psnr_list = []
            ssim_list = []
            start_time = time.time()
            train_bar = tqdm(train_loader)
            for i, (feature, label) in enumerate(train_bar):
                label, feature = label.to(device), feature.to(device)  # 将标签、特征都放入设备中
                optimizer.zero_grad()
                predicted = model(feature)  # 获得模型输出

                # print(len(predicted))
                # print(predicted[0].shape)

                match opt.loss:  # 计算loss n
                    case 'MaskedMSE':
                        loss = criterion(feature, label, predicted)
                    case _:
                        loss = criterion(predicted, label)
                loss.backward()  # 反向传播，计算梯度
                psnr = batch_PSNR(predicted, label, torch.max(label))  # 计算psnr n
                if opt.dtype == '5d':
                    B, C, D1, D2, D3, D4, D5 = predicted.shape
                    predicted = predicted.view(B, C, D1, D2 * D3, D4 * D5)
                    label = label.view(B, C, D1, D2 * D3, D4 * D5)
                ssim = SSIM(predicted, label).item()  # 计算ssim
                # predicted = predicted.view(B, C, D1, D2, D3, D4, D5)
                # label = label.view(B, C, D1, D2, D3, D4, D5)
                optimizer.step()  # 更新模型参数 n
                scheduler.step()

                if torch.isnan(loss):
                    raise ValueError(f"Epoch {epoch}, Batch {i}: Loss is NaN, stopping training.")

                # 获得结果 n
                loss_list.append(loss.item())
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                # print("[Epoch %d][%d/%d] Loss: %.4f PSNR: %.4f" %
                #       (epoch + 1, i + 1, len(train_loader), loss.item(), psnr))
                train_bar.set_description(
                    desc='[%d/%d] Loss: %.10f\tPSNR: %.2f\tSSIM: %.4f' % (
                    epoch + 1, opt.epochs, loss.item(), psnr, ssim))
                # train_bar.set_description(
                #     desc='[%d/%d] Loss: %.6f PSNR: %.2f' % (
                #     epoch + 1, opt.epochs, loss.item(), psnr))

            epoch_loss = sum(loss_list) / len(loss_list)
            epoch_psnr = sum(psnr_list) / len(psnr_list)
            epoch_ssim = sum(ssim_list) / len(ssim_list)
            print("\n[Epoch %d] Training Avg Loss: %.10f\tAvg PSNR: %.2f\tSSIM: %.4f" %
                  (epoch + 1, epoch_loss, epoch_psnr, epoch_ssim))
            # print("\n[Epoch %d] Training Avg Loss: %.6f Avg PSNR: %.2f" %
            #       (epoch + 1, epoch_loss, epoch_psnr))
            results_train['Loss'].append(epoch_loss)
            results_train['PSNR'].append(epoch_psnr)
            results_train['SSIM'].append(epoch_ssim)

            # 验证
            model.eval()
            if val_loader != None:
                with torch.no_grad():
                    val_bar = tqdm(val_loader)
                    loss_list = []
                    psnr_list = []
                    ssim_list = []
                    for i, (feature, label) in enumerate(val_bar):
                        label, feature = label.to(device), feature.to(device)  # 将标签、特征都放入设备中

                        predicted = model(feature)  # 获得模型输出

                        match opt.loss:
                            case 'MaskedMSE':
                                loss = criterion(feature, label, predicted)
                            case _:
                                loss = criterion(predicted, label)

                        psnr = batch_PSNR(predicted, label, torch.max(label))  # 计算psnr
                        if opt.dtype == '5d':
                            B, C, D1, D2, D3, D4, D5 = predicted.shape
                            predicted = predicted.view(B, C, D1, D2 * D3, D4 * D5)
                            label = label.view(B, C, D1, D2 * D3, D4 * D5)
                        ssim = SSIM(predicted, label).item()
                        # 获得结果
                        loss_list.append(loss.item())
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        # print("[Epoch %d][%d/%d] Loss: %.4f PSNR: %.4f" %
                        #       (epoch + 1, i + 1, len(val_loader), loss.item(), psnr))
                        val_bar.set_description(
                            desc='[%d/%d] Loss: %.10f\tPSNR: %.2f\tSSIM: %.4f' % (
                                epoch + 1, opt.epochs, loss.item(), psnr, ssim))
                        # val_bar.set_description(
                        #     desc='[%d/%d] Loss: %.4f PSNR: %.4f' % (
                        #         epoch + 1, opt.epochs, loss.item(), psnr))
                    epoch_loss = sum(loss_list) / len(loss_list)
                    epoch_psnr = sum(psnr_list) / len(psnr_list)
                    epoch_ssim = sum(ssim_list) / len(ssim_list)
                    print("\n[Epoch %d] Validating Avg Loss: %.10f\tAvg PSNR: %.2f\tSSIM: %.4f" %
                          (epoch + 1, epoch_loss, epoch_psnr, epoch_ssim))
                    # print("\n[Epoch %d] Validating Avg Loss: %.4f Avg PSNR: %.2f" %
                    #       (epoch + 1, epoch_loss, epoch_psnr))
                    # 存储验证集SNR和loss
                    results_val['Loss'].append(epoch_loss)
                    results_val['PSNR'].append(epoch_psnr)
                    results_val['SSIM'].append(epoch_ssim)
                metric = epoch_psnr * epoch_ssim
                # metric = epoch_psnr
                if metric > best:  # 记录效果最好的模型为recon_best
                    best = metric
                    if not opt.cp:
                        os.makedirs(save_dir, exist_ok=True)

                    torch.save(model.state_dict(),
                        os.path.join(save_dir if not opt.cp else read_save_dir,
                        'recon_best.pth'))
            # 保存模型
            if not opt.cp:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(),
                os.path.join(save_dir if not opt.cp else read_save_dir,
                f'recon{epoch + 1}.pth'))
            # 保存检查点
            save_checkpoint(model, optimizer, scheduler, epoch, timelist, results_train, results_val,
                os.path.join(save_dir if not opt.cp else read_save_dir, f'checkpoint.pth'))

            # scheduler.step()
            torch.cuda.empty_cache()
            end_time = time.time()
            timelist.append(end_time - start_time)
            print(f"[Epoch {epoch + 1}] Total time used: {end_time - start_time}s")


    except ValueError as e:
        print(e)
        # 释放GPU资源
        del feature, label, predicted, loss
        torch.cuda.empty_cache()
        gc.collect()

    with open(os.path.join(save_dir if not opt.cp else read_save_dir,
                           'training_info.log'), 'w') as f:
        f.write("training times:\n")
        for line in timelist:
            f.write(str(line) + "\n")
        f.write("avg_train_loss:\n")
        for line in results_train["Loss"]:
            f.write(str(line) + "\n")
        f.write("avg_train_psnr:\n")
        for line in results_train["PSNR"]:
            f.write(str(line) + "\n")
        f.write("avg_train_ssim:\n")
        for line in results_train["SSIM"]:
            f.write(str(line) + "\n")
        f.write("avg_val_loss:\n")
        for line in results_val["Loss"]:
            f.write(str(line) + "\n")
        f.write("avg_val_psnr:\n")
        for line in results_val["PSNR"]:
            f.write(str(line) + "\n")
        f.write("avg_val_ssim:\n")
        for line in results_val["SSIM"]:
            f.write(str(line) + "\n")


    # 绘制第一个图：训练损失
    # print(len(avg_loss_list))
    plt.figure(figsize=(8, 6))
    plt.plot(results_train["Loss"])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(save_dir if not opt.cp else read_save_dir,
                             'training_loss.png'))
    plt.close()  # 关闭当前图形，准备绘制下一个

    # 绘制第二个图：训练SNR
    plt.figure(figsize=(8, 6))
    plt.plot(results_train["PSNR"])
    plt.title('Training PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(save_dir if not opt.cp else read_save_dir,
                             'training_psnr.png'))
    plt.close()  # 关闭当前图形

# 创建必要文件夹
os.makedirs('./logs', exist_ok=True)
os.makedirs('./results', exist_ok=True)
# current_time在程序一开始便记录了
log_filename = f'./logs/{opt.net}_{current_time}.log'
# 设置日志记录器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename)
                        , logging.StreamHandler()
                              ])

# 创建一个日志记录器
logger = logging.getLogger()
# 将标准输出和标准错误重定向到日志记录器，同时保留原始的输出到控制台
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''

    def write(self, message):
        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.level(line + '\n')

    def flush(self):
            if self.buffer != '':
                self.level(self.buffer.strip())
                self.buffer = ''

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

if __name__ == "__main__":
    main()



