# Code Implementation of the training process
# you can change the parameters here or in the shell scripts (recommended)
import os, glob, argparse, shutil, random
import torch
import ast


from utils import *
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description="Reconstruction Test")
parser.add_argument("--mtype", type=str, default='regular', help='missing type; regular: delete one out of every three; irregular: randomly delete 50% of traces')
parser.add_argument("--mrate", type=float, default='0.0', help='missing rate; applicable only when --mtype is irregular')
parser.add_argument("--urf", type=int, default='0', help='uniform random factor')
parser.add_argument("--folder", type=str, default='./data/test_3d', help='test data folder')
parser.add_argument("--epochs", type=int, default=0, help='epochs')
parser.add_argument("--net", type=str, default='LMGM', help="network")
parser.add_argument("--gpunum", type=int, default=1, help="Number of GPUs")
parser.add_argument("--shape", type=str, default='(32, 32, 32)', help="Size of patch")
parser.add_argument("--testtype", type=int, default=1, help="0: directly testing; 1: testing with sliding window")
parser.add_argument("--mfolder", type=str, default='0000', help="**Must: the folder of the trained model")
parser.add_argument("--thre", type=int, default=100, help="")
parser.add_argument("--xmiss", type=bool, default=False, help="")
parser.add_argument("--tl", type=bool, default=False, help="")
parser.add_argument("--tlfolder", type=str, default='', help="")
parser.add_argument("--ntype", type=str, default='3d', help="network type: 2d, 3d, or tl")
parser.add_argument("--norm", type=str, default="absmax", help="normalization of the data")

opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 加载模型
    print('Loading model ...\n')
    # 导入网络的模块
    from importlib import import_module

    match opt.ntype:
        case '3d':
            net_folder = "src.networks."
        # case 'tl':
        #     net_folder = "transfer_learning.networks."
        # case '2d':
        #     net_folder = "networks.2d."
    module_name = net_folder + opt.net
    net_module = import_module(module_name)
    net_class = getattr(net_module, opt.net)
    net = net_class()



    # 采用多GPU进行预测
    gpu_num = opt.gpunum
    device_ids = multi_gpu(gpu_num, opt.thre)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 将 device_ids 转换为逗号分隔的字符串
    cuda_visible_devices = ",".join(str(id) for id in device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(net, device_ids=device_ids)
    # model = net

    # if not opt.tl:
    #     if opt.ntype == "3d":
    #         model_folder = "3d"
    #     elif opt.ntype == "2d":
    #         model_folder = "2d"
    # else:
    #     model_folder = opt.tlfolder
    model_folder = "3d"

    save_dir = f'./trained_models/{model_folder}/{opt.mfolder}'
    model_filename = 'recon_best.pth' if opt.epochs == 0 else f'recon{opt.epochs}.pth'
    # model.load_state_dict(torch.load(os.path.join(save_dir, model_filename)))
    model.load_state_dict(torch.load(os.path.join(save_dir, model_filename), map_location="cpu"))
    model = model.to(device)

    model.eval()  # 评估模式
    print('Loading data info ...\n')
    # files_source = glob.glob(os.path.join('./sdata/test_recon', 'field_label.mat'))
    files_source = glob.glob(os.path.join(f'./data/{opt.folder}/test', '*.mat'))
    assert len(files_source) != 0
    with torch.no_grad():
        # 处理数据
        for f in files_source:

            fname = os.path.basename(f)
            start_time = time.time()
            print(f"\nFile \'{os.path.basename(f)}\' starting random deleting and reconstructing")
            # 读取mat数据，以及其中的数组
            mat_data = sio.loadmat(f)
            arr = mat_data[os.path.basename(f)[:-4]]
            print("Original data: ", arr.shape)
            # 取前几个测线
            block_shape = ast.literal_eval(opt.shape)
            print("Training block:", block_shape)
            print("Data to be tested:", arr.shape)

            n_times, n_inline, n_xline = arr.shape
            zero_mask = (arr == 0)

            arr_max = np.max(np.abs(arr))

            # print(f'arr_max: {arr_max}')
            miss_arr = np.copy(arr)
            print(miss_arr.shape)
            print(f"Missing rate: {opt.mrate}")
            # 模拟不同类型的缺失
            if opt.mtype == 'regular':  # 常规缺失：每2道删除1道
                match opt.mrate:
                    case 0.33:
                        for il in range(1, n_inline, 3):
                            miss_arr[:, il, :] = 0
                    case 0.5:
                        for il in range(1, n_inline, 2):
                            miss_arr[:, il, :] = 0
                    case 0.67:
                        for il in range(1, n_inline, 3):
                            miss_arr[:, il:il + 2, :] = 0
                    case 0.75:
                        for il in range(1, n_inline, 4):
                            miss_arr[:, il:il + 3, :] = 0
                    case 0.8:
                        for il in range(1, n_inline, 5):
                            miss_arr[:, il:il + 4, :] = 0
                    case 0.83:
                        for il in range(1, n_inline, 6):
                            miss_arr[:, il:il + 5, :] = 0
                    case 0.86:
                        for il in range(1, n_inline, 7):
                            miss_arr[:, il:il + 6, :] = 0
                    case 0.88:
                        for il in range(1, n_inline, 8):
                            miss_arr[:, il:il + 7, :] = 0
                    case 0.89:
                        for il in range(1, n_inline, 9):
                            miss_arr[:, il:il + 8, :] = 0
                    case 0.9:
                        for il in range(1, n_inline, 10):
                            miss_arr[:, il:il + 9, :] = 0
                    case _:
                        mrate = [0.33, 0.5, 0.67, 0.75, 0.8, 0.83, 0.86, 0.88, 0.89, 0.9]
                        raise ValueError(f"{opt.mrate} is an unsupported missing rate. \n"
                                         f"Supported rates are: {mrate}")

                mask = miss_arr != 0

            elif opt.mtype == 'irregular':  # 非常规缺失：随机删除道数。为了复现，人工设置随机种子
                seed_dict = []
                for i in range(arr.shape[2]):
                    # seed_dict.append((i + 1) * 666) # 以前的种子
                    seed_dict.append((i + 1) * 777)

                print(seed_dict)
                miss_arr, mask = remove_traces_3d(arr, opt.mrate, opt.mrate,
                                         'ur', ur_factor=opt.urf, seed_dict=seed_dict)


            elif opt.mtype == 'none':
                miss_arr = arr.copy()
                mask = miss_arr != 0


            # 归一化
            match opt.norm:
                case 'absmax':
                    arr = arr / arr_max
                    miss_arr = miss_arr / arr_max
                case 'log1p':
                    arr = norm_by_log(arr)
                    miss_arr = norm_by_log(miss_arr)
                case 'TPNN':
                    arr, arr_mean, arr_std = norm_by_arcsinh_std(arr)
                    miss_arr, _, _ = norm_by_arcsinh_std(miss_arr, arr_mean, arr_std)
                case _:
                    pass

            if opt.testtype == 0:  # 直接输入整幅，适用于CNN-based
                if "UNet" in opt.net:
                    arr = pad_3d_reflect(arr, 32, 32, 32)
                    miss_arr = pad_3d_reflect(miss_arr, 32, 32, 32)
                    print(f"After padding: {arr.shape}")
                with torch.no_grad():  # this can save much memory
                    miss_arr = torch.tensor(miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    reconstructed = model(miss_arr).cpu().detach().numpy().squeeze()
                miss_arr = miss_arr.cpu().detach().numpy().squeeze()

                match opt.norm:
                    case 'absmax':
                        arr = arr * arr_max
                        miss_arr = miss_arr * arr_max
                        reconstructed = reconstructed * arr_max
                    case 'log1p':
                        arr = denorm_by_log(arr)
                        miss_arr = denorm_by_log(miss_arr)
                        reconstructed = denorm_by_log(reconstructed)
                    case 'TPNN':
                        arr = denorm_by_arcsinh_std(arr, arr_mean, arr_std)
                        miss_arr = denorm_by_arcsinh_std(miss_arr, arr_mean, arr_std)
                        reconstructed = denorm_by_arcsinh_std(reconstructed, arr_mean, arr_std)
                    case _:
                        pass

                arr = arr[:n_times, :n_inline, :n_xline]
                miss_arr = miss_arr[:n_times, :n_inline, :n_xline]
                reconstructed = reconstructed[:n_times, :n_inline, :n_xline]



            elif opt.testtype == 1:  # (平滑处理)分片预测，然后结合

                # 对数据进行填充至能整除训练块大小
                arr = pad_3d_reflect(arr, *block_shape)
                miss_arr = pad_3d_reflect(miss_arr, *block_shape)
                print("After padding: ", arr.shape)

                # CNN_based = ['MIR_3d']
                # CNN_based = ['UNet_3d']
                CNN_based = []
                if opt.net not in CNN_based:
                    with torch.no_grad():  # this can save much memory
                        print("Predicting...")
                        reconstructed = test_with_sliding_window_3d(
                            miss_arr, block_shape, model, device, tuple(x // 2 for x in block_shape))
                    print(f"reconstructed.shape: {reconstructed.shape}")
                else:
                    miss_arr_t = torch.tensor(miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    reconstructed = model(miss_arr_t).cpu().detach().numpy().squeeze()
                    # reconstructed = np.zeros(arr.shape)
                    # for i in range(0, n_inline, block_shape[1]):
                    #     sub_miss_arr = miss_arr[:, i:i + block_shape[1], :]
                    #     print(f"Predicting sub_miss_arr: [:, {i}:{i + block_shape[1]}, :]")
                    #     sub_miss_arr = torch.tensor(sub_miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    #     recon = model(sub_miss_arr).cpu().detach().numpy().squeeze()
                    #     reconstructed[:, i:i + block_shape[1], :] = recon

                match opt.norm:
                    case 'absmax':
                        arr = arr * arr_max
                        miss_arr = miss_arr * arr_max
                        reconstructed = reconstructed * arr_max
                    case 'log1p':
                        arr = denorm_by_log(arr)
                        miss_arr = denorm_by_log(miss_arr)
                        reconstructed = denorm_by_log(reconstructed)
                    case 'TPNN':
                        arr = denorm_by_arcsinh_std(arr, arr_mean, arr_std)
                        miss_arr = denorm_by_arcsinh_std(miss_arr, arr_mean, arr_std)
                        reconstructed = denorm_by_arcsinh_std(reconstructed, arr_mean, arr_std)
                    case _:
                        pass

                arr = arr[:n_times, :n_inline, :n_xline]
                miss_arr = miss_arr[:n_times, :n_inline, :n_xline]
                reconstructed = reconstructed[:n_times, :n_inline, :n_xline]

            elif opt.testtype == 2:  # 取其中一小片进行预测
                print(arr.shape)
                arr = arr[142:174, 169:201, 84:116]
                print(arr.shape)
                assert arr.shape == (opt.win, opt.win, opt.win)
                miss_arr = np.copy(arr)

                if opt.mtype == 'regular':  # 常规缺失：每2道删除1道；每2条线删除1条
                    for xline in range(0, miss_arr.shape[2], 3):
                        miss_arr[:, :, xline] = 0
                    for line in range(0, miss_arr.shape[0], 3):
                        miss_arr[line, :, :] = 0
                elif opt.mtype == 'irregular':  # 非常规缺失：随机删除道数。为了复现，人工设置随机种子
                    random.seed(98438)  # 98438： 256道合成数据、kerry数据的均匀种子；903： 拖缆数据的均匀种子；
                    del_num1 = int(miss_arr.shape[2] * 0.5)  # 删除道
                    del_num2 = int(miss_arr.shape[0] * 0.5)  # 删除测线
                    del_xline = random.sample(range(miss_arr.shape[2]), del_num1)
                    del_inline = random.sample(range(miss_arr.shape[0]), del_num2)
                    # print(sorted(del_cols))
                    for xline in del_xline:
                        miss_arr[:, :, xline] = 0
                    for line in del_inline:
                        miss_arr[line, :, :] = 0
                miss_t = torch.tensor(miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():  # this can save much memory
                    reconstructed = model(miss_t).cpu().detach().numpy().squeeze()

                arr = (arr * arr_max)[:n_times, :n_inline, :n_xline]
                miss_arr = (miss_arr * arr_max)[:n_times, :n_inline, :n_xline]
                reconstructed = (reconstructed * arr_max)[:n_times, :n_inline, :n_xline]

            elif opt.testtype == 3:  # 用二维模型测试3D数据
                # (平滑处理)分片预测，然后结合

                # 对数据进行填充至能整除训练块大小
                arr = pad_3d(arr, block_shape[0], block_shape[1], block_shape[2])
                miss_arr = pad_3d(miss_arr, block_shape[0], block_shape[1], block_shape[2])
                print("After padding: ", arr.shape)

                # CNN_based = ['MIR_3d']
                CNN_based = ['UNet_3d']
                # CNN_based = []
                if opt.net not in CNN_based:
                    with torch.no_grad():  # this can save much memory
                        print("Predicting...")

                        reconstructed = test_with_sliding_window_2d_from_3d(
                            miss_arr, block_shape, model, device, tuple(x // 2 for x in block_shape))
                    print(f"reconstructed.shape: {reconstructed.shape}")
                else:
                    miss_arr_t = torch.tensor(miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    reconstructed = model(miss_arr_t).cpu().detach().numpy().squeeze()
                    # reconstructed = np.zeros(arr.shape)
                    # for i in range(0, n_inline, block_shape[1]):
                    #     sub_miss_arr = miss_arr[:, i:i + block_shape[1], :]
                    #     print(f"Predicting sub_miss_arr: [:, {i}:{i + block_shape[1]}, :]")
                    #     sub_miss_arr = torch.tensor(sub_miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    #     recon = model(sub_miss_arr).cpu().detach().numpy().squeeze()
                    #     reconstructed[:, i:i + block_shape[1], :] = recon

                match opt.norm:
                    case 'absmax':
                        arr = arr * arr_max
                        miss_arr = miss_arr * arr_max
                        reconstructed = reconstructed * arr_max
                    case 'log1p':
                        arr = denorm_by_log(arr)
                        miss_arr = denorm_by_log(miss_arr)
                        reconstructed = denorm_by_log(reconstructed)
                    case _:
                        pass

                arr = arr[:n_times, :n_inline, :n_xline]
                miss_arr = miss_arr[:n_times, :n_inline, :n_xline]
                reconstructed = reconstructed[:n_times, :n_inline, :n_xline]

            else:
                raise NotImplementedError(f"testtype {opt.testtype} not implemented.")

            # 进行回填操作
            print(mask.min(), mask.max(), mask.mean())
            reconstructed, difference = refill(arr=arr,
                                               reconstructed=reconstructed,
                                               mask=mask)
            reconstructed = reconstructed * (1 - zero_mask)


            # 保存为mat文件
            result_dir = f'./results/{model_folder}/{opt.net}/{os.path.basename(opt.folder)}/'
            os.makedirs(result_dir, exist_ok=True)
            sio.savemat(os.path.join(result_dir, f'{os.path.basename(f)[:-4]}.mat'),
                        {'complete': arr, 'missing': miss_arr,
                         'reconstructed': reconstructed, 'difference': reconstructed - arr})

            # 衡量数值性能
            snr_miss = array_SNR(miss_arr, arr)
            psnr_miss = array_PSNR(miss_arr, arr)
            rmse_miss = array_RMSE(miss_arr, arr)


            snr_test = array_SNR(reconstructed, arr)
            psnr_test = array_PSNR(reconstructed, arr)
            rmse_test = array_RMSE(reconstructed, arr)

            shape = np.array(arr.shape)
            candidates = np.where(shape < 7)[0]
            if len(candidates) == 0:
                ssim_miss, _ = ssim(miss_arr, arr, full=True, data_range=arr_max)
                ssim_test, _ = ssim(reconstructed, arr, full=True, data_range=arr_max)
            else:
                dim = candidates[np.argmin(shape[candidates])]
                size = arr.shape[dim]
                odd = size - 1 if size % 2 == 0 else size - 2
                odd = max(odd, 1)
                ssim_miss, _ = ssim(miss_arr, arr, full=True, data_range=arr_max, win_size=odd)
                ssim_test, _ = ssim(reconstructed, arr, full=True, data_range=arr_max, win_size=odd)

            end_time = time.time()
            print(f"Inference time of {fname}: {end_time - start_time}s")

            with open(os.path.join(result_dir, f'metrics_{opt.folder}_interpolation.log'), 'w') as result_file:
                print(f"SNR on missing {os.path.basename(f)}: {snr_miss: .2f}")
                print(f"PSNR on missing {os.path.basename(f)}: {psnr_miss: .2f}")
                print(f"RMSE on missing {os.path.basename(f)}:  {rmse_miss: .4f}")
                print(f"SSIM on missing {os.path.basename(f)}: {ssim_miss: .6f}")
                print(f"SNR on reconstructed {os.path.basename(f)}: {snr_test: .2f}")
                print(f"PSNR on reconstructed {os.path.basename(f)}: {psnr_test: .2f}")
                print(f"RMSE on reconstructed {os.path.basename(f)}:  {rmse_test: .4f}")
                print(f"SSIM on reconstructed {os.path.basename(f)}: {ssim_test: .6f}")
                result_file.write(f"SNR on missing {os.path.basename(f)}:  {snr_miss: .2f}\n")
                result_file.write(f"PSNR on missing {os.path.basename(f)}:  {psnr_miss: .2f}\n")
                result_file.write(f"RMSE on missing {os.path.basename(f)}:  {rmse_miss: .4f}\n")
                result_file.write(f"SSIM on missing {os.path.basename(f)}: {ssim_miss: .6f}\n")
                result_file.write(f"SNR on reconstructed {os.path.basename(f)}:  {snr_test: .2f}\n")
                result_file.write(f"PSNR on reconstructed {os.path.basename(f)}:  {psnr_test: .2f}\n")
                result_file.write(f"RMSE on reconstructed {os.path.basename(f)}:  {rmse_test: .4f}\n")
                result_file.write(f"SSIM on reconstructed {os.path.basename(f)}: {ssim_test: .6f}\n")

                print("----------------------------------------------------------")
                for i in range(0, arr.shape[2]):
                    arr_i = arr[:, :, i]
                    miss_arr_i = miss_arr[:, :, i]
                    reconstructed_i = reconstructed[:, :, i]

                    snr_miss = array_SNR(miss_arr_i, arr_i)
                    psnr_miss = array_PSNR(miss_arr_i, arr_i)
                    rmse_miss = array_RMSE(miss_arr_i, arr_i)

                    snr_test = array_SNR(reconstructed_i, arr_i)
                    psnr_test = array_PSNR(reconstructed_i, arr_i)
                    rmse_test = array_RMSE(reconstructed_i, arr_i)

                    shape = np.array(arr_i.shape)
                    candidates = np.where(shape < 7)[0]
                    if len(candidates) == 0:
                        ssim_miss, _ = ssim(miss_arr_i, arr_i, full=True, data_range=arr_max)
                        ssim_test, _ = ssim(reconstructed_i, arr_i, full=True, data_range=arr_max)
                    else:
                        dim = candidates[np.argmin(shape[candidates])]
                        size = arr_i.shape[dim]
                        odd = size - 1 if size % 2 == 0 else size - 2
                        odd = max(odd, 1)
                        ssim_miss, _ = ssim(miss_arr_i, arr_i, full=True, data_range=arr_max, win_size=odd)
                        ssim_test, _ = ssim(reconstructed_i, arr_i, full=True, data_range=arr_max, win_size=odd)
                    print(f"SNR on missing {os.path.basename(f)}, slice {i}:  {snr_miss: .2f}")
                    print(f"PSNR on missing {os.path.basename(f)}, slice"
                          f" {i}:  {psnr_miss: .2f}")
                    print(f"RMSE on missing {os.path.basename(f)}, slice {i}:  {rmse_miss: .4f}")
                    print(f"SSIM on missing {os.path.basename(f)}, slice {i}: {ssim_miss: .6f}")
                    print(f"SNR on reconstructed {os.path.basename(f)}, slice {i}:  {snr_test: .2f}")
                    print(f"PSNR on reconstructed {os.path.basename(f)}, slice {i}:  {psnr_test: .2f}")
                    print(f"RMSE on reconstructed {os.path.basename(f)}, slice {i}:  {rmse_test: .4f}")
                    print(f"SSIM on reconstructed {os.path.basename(f)}, slice {i}: {ssim_test: .6f}")
                    print()
                    result_file.write(f"SNR on missing {os.path.basename(f)}, slice {i}:  {snr_miss: .2f}\n")
                    result_file.write(f"PSNR on missing {os.path.basename(f)}, slice {i}:  {psnr_miss: .2f}\n")
                    result_file.write(f"RMSE on missing {os.path.basename(f)}, slice {i}:  {rmse_miss: .4f}\n")
                    result_file.write(f"SSIM on missing {os.path.basename(f)}, slice {i}: {ssim_miss: .6f}\n")
                    result_file.write(f"SNR on reconstructed {os.path.basename(f)}, slice {i}:  {snr_test: .2f}\n")
                    result_file.write(f"PSNR on reconstructed {os.path.basename(f)}, slice {i}:  {psnr_test: .2f}\n")
                    result_file.write(f"RMSE on reconstructed {os.path.basename(f)}, slice {i}:  {rmse_test: .4f}\n")
                    result_file.write(f"SSIM on reconstructed {os.path.basename(f)}, slice {i}: {ssim_test: .6f}\n")
                print("----------------------------------------------------------")



if __name__ == "__main__":
    main()
