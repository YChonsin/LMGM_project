import argparse
import subprocess
import torch
import torch.nn as nn
import numpy as np
# from skimage.measure.simple_metrics import compare_psnr
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity as ssim
import random, os, time
import h5py
import scipy.io as sio
from torchmetrics.functional import structural_similarity_index_measure as SSIM

def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def batch_PSNR(predict, target, max_value=1.):
    mse = torch.mean((predict - target) ** 2)
    psnr = 10 * torch.log10((max_value ** 2) / mse)
    return psnr.item()


def array_SNR(prediction, target):
    """
    计算处理后数据的信噪比（SNR）。

    :param target: 2D numpy array，表示原始信号数据
    :param prediction: 2D numpy array，表示处理后的数据
    :return: 信噪比，以分贝（dB）为单位
    """
    # 计算信号的能量
    signal_energy = np.sum(np.square(target))

    # 插值任务中，噪声是插值后数据与原始数据之差的能量
    noise = prediction - target
    noise_energy = np.sum(np.square(noise))

    # 计算信噪比 (SNR)
    snr = 10 * np.log10(signal_energy / noise_energy)

    return snr

def fill_zero_with_inf(data, inf):
    # 构建 mask
    mask = (data != 0).astype(np.uint8)

    # 计算非零区域平均值
    non_zero_values = data[data != 0]
    mean_value = np.mean(non_zero_values)

    print(f"非零区域平均值: {mean_value}")

    # 填充
    filled_data = data.copy()
    filled_data[data == 0] = inf

    return filled_data, mask

def recover_zero_with_inf(filled_data, mask):
    # 构建 mask
    data = filled_data * mask
    return data



def array_PSNR(prediction, target):
    """
    计算峰值信噪比（PSNR）。

    :param original_image: 2D numpy array，表示原始图像
    :param processed_image: 2D numpy array，表示处理后的图像
    :return: 峰值信噪比（PSNR），单位为dB
    """
    # 计算均方误差（MSE）
    mse = np.mean((target - prediction) ** 2)

    if mse == 0:
        return float('inf')  # 如果 MSE 为 0，则 PSNR 无穷大，表示没有噪声

    # 对于 8 位图像，MAX=255
    MAX = np.max(target)

    # 计算PSNR
    psnr = 10 * np.log10((MAX ** 2) / mse)

    return psnr
def NRMSE(predict, target):
    rmse = torch.sqrt(torch.mean((predict - target) ** 2))
    nrmse = rmse / torch.std(target)
    return nrmse.item()


# def batch_PSNR(img, imclean, data_range):
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     PSNR = 0
#     for i in range(Img.shape[0]):
#         PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
#     return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


def array_RMSE(predict, target):
    """
    计算均方根误差（RMSE）。

    :param target: 2D numpy array，表示原始信号数据
    :param predict: 2D numpy array，表示处理后的数据或预测数据
    :return: 均方根误差（RMSE）
    """
    # 计算误差（处理后数据与原始数据之差）
    error = predict - target

    # 计算均方根误差
    mse = np.mean(np.square(error))
    rmse = np.sqrt(mse)

    return rmse

def array_NRMSE(predict, target):
    return array_RMSE(predict, target) / (np.max(target) - np.min(target))

# def recon_result_subtraction(mat):
#     return mat['reconstructed'] - mat['complete']

def multi_gpu(gpu_num, thre=100):
    '''

    :param gpu_num: 所需GPU的数量
    :return:  selected_gpus: 可用GPU的ID列表，如果为空列表则无可用
    '''
    selected_gpus = []
    if torch.cuda.is_available():
        # 获取可用的GPU数量
        available_gpus = torch.cuda.device_count()

        used_gpus = []
        # 使用nvidia-smi获取正在使用的 GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE)
        gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
        gpu_memory = [int(x) for x in gpu_memory]

        # 如果该GPU的显存占用大于100M，则认为被使用了
        for i in range(available_gpus):
            if gpu_memory[i] > thre:
                used_gpus.append(i)

        free_gpus = [i for i in range(available_gpus) if i not in used_gpus]

        if gpu_num > len(free_gpus):
            print(f"Requested {gpu_num} GPUs, but only {len(free_gpus)} are available.")
            return []

        print(f"Number of available GPUs: {len(free_gpus)}")
        # 选择空闲的 GPU
        selected_gpus = free_gpus[:gpu_num]
        print(f"Available GPUs: {selected_gpus}")

    if len(selected_gpus) == 0:
        raise ValueError("No GPU devices available")
    return selected_gpus


def remove_traces(arr, low=0.1, high=0.3, misstype='r',
                  zero_value = 0., ur_factor=4, seed=-1):
    '''
    第一个维度：时间采样点  第二个维度：道数
    :param arr: 2d numpy array
    :param low:
    :param high:
    :param misstype: 'r' for random, 'p' for pattern, 'c' for consecutive
    :return:
    '''
    arr = np.copy(arr)
    rows, cols = arr.shape
    if low > high:
        raise ValueError('low must be less than high')
    if low >= 0. and high < 1.:
        del_num = int(random.uniform(low, high) * cols)
    elif low == high:
        del_num = int(low * cols)
    elif low >= 1:
        del_num = random.randrange(low, high)

    match misstype:
        case 'r':
            del_cols = random.sample(range(cols), del_num)
            for col in del_cols:
                arr[:, col] = zero_value

        case 'p':
            step = cols // del_num
            for i in range(del_num):
                arr[:, i * step] = zero_value

        case 'c':
            start_col = random.randrange(0, cols - del_num)
            arr[:, start_col:start_col + del_num] = zero_value

        case 'ur':  # randomly deleting in a uniform way
            if seed != -1:
                random.seed(seed)
                print(f"Using seed {seed}")
            for i in range(0, cols, ur_factor):
                subset = ur_factor if cols - i > ur_factor else cols - i
                sub_del_num = int(random.uniform(low, high) * subset)
                sub_del_cols = random.sample(range(subset), sub_del_num)
                # print(subset, sub_del_num, sub_del_cols)

                for col in sub_del_cols:
                    arr[:, i + col] = zero_value
        case _:
            raise NotImplementedError('Unknown misstype')

    return arr

def remove_traces_3d(arr, low=0.1, high=0.3, misstype='r',
                     zero_value = 0., ur_factor=4, seed_dict=None):
    '''
    第一个维度：时间采样点  第二个维度：道数
    :param arr: 2d numpy array
    :param low:
    :param high:
    :param misstype: 'r' for random, 'p' for pattern, 'c' for consecutive
    :return:
    '''
    arr = np.copy(arr)
    sample, n_xline, n_inline = arr.shape
    if low > high:
        raise ValueError('low must be less than high')

    match misstype:
        case 'r':
            del_num = int(random.uniform(low, high) * n_xline * n_inline)
            del_traces_idx = random.sample(range(n_xline * n_inline), del_num)
            arr = arr.reshape(sample, n_xline * n_inline)
            mask = np.ones((sample, n_xline * n_inline))
            for i in del_traces_idx:
                mask[:, i] = zero_value
            arr = arr * mask
            arr = arr.reshape(sample, n_xline, n_inline)

        case 'ur':  # randomly deleting in a uniform way
            mask = np.ones((sample, n_xline, n_inline))
            for idx in range(n_inline):
                if seed_dict is not None:
                    random.seed(seed_dict[idx])
                    print(f"Using seed {seed_dict[idx]}")
                for i in range(0, n_xline, ur_factor):
                    subset = ur_factor if n_xline - i > ur_factor else n_xline - i
                    sub_del_num = int(random.uniform(low, high) * subset)
                    # print(sub_del_num)
                    sub_dels = random.sample(range(subset), sub_del_num)
                    # print(sub_dels)
                    for sub_idx in sub_dels:
                        mask[:, i + sub_idx, idx] = zero_value

            arr = arr * mask

            #
            # total_traces = n_inline * n_xline
            # arr = arr.reshape(sample, total_traces)
            # # print(arr.shape)
            # for i in range(0, total_traces, ur_factor):
            #     subset = ur_factor if total_traces - i > ur_factor else total_traces - i
            #     # print(subset)
            #     sub_del_num = int(random.uniform(low, high) * subset)
            #     # print(sub_del_num)
            #     sub_dels = random.sample(range(subset), sub_del_num)
            #     # print(sub_dels)
            #     for idx in sub_dels:
            #         arr[:, i + idx] = zero_value
            #
            # arr = arr.reshape(sample, n_inline, n_xline)

        case _:
            raise NotImplementedError('Unknown misstype')

    return arr, mask

def remove_traces_5d(arr, low=0.1, high=0.3, misstype='r',
                  zero_value = 0., seed=0):
    '''
    第一个维度：时间采样点  第二个维度：道数
    :param arr: 2d numpy array
    :param low:
    :param high:
    :param misstype: 'r' for random,
    :return:
    '''
    if low > high:
        raise ValueError('low must be less than high')

    arr = np.copy(arr)
    ts, D1, D2, D3, D4 = arr.shape
    if low > high:
        raise ValueError('low must be less than high')

    match misstype:
        case 'r':
            total_traces = D1 * D2 * D3 * D4
            del_num = int(random.uniform(low, high) * total_traces)
            if seed != 0:
                random.seed(seed)
            dels = random.sample(range(total_traces), del_num)
            arr = arr.reshape(ts, total_traces)
            for i in dels:
                arr[:, i] = zero_value
            arr = arr.reshape(ts, D1, D2, D3, D4)

        case _:
            raise NotImplementedError('Unknown misstype')

    return arr



# def remove_traces_3d(arr, low=0.1, high=0.3,
#                      misstype='r', zero_value = 0., direction = 'None'):
#     '''
#     第一个维度：sample  第二个维度：inline  第三个维度：xline
#     :param arr: 2d numpy array
#     :param low:
#     :param high:
#     :param misstype: 'r' for random, 'p' for pattern, 'c' for consecutive
#     :return:
#     '''
#     arr = np.copy(arr)
#     sample, inline, xline = arr.shape
#     if low >= high:
#         raise ValueError('low must be less than high')
#
#     if misstype == 'r':
#         if direction == 'inline':
#             del_num = int(random.uniform(low, high) * inline)
#             del_cols = random.sample(range(inline), del_num)
#             for line in del_cols:
#                 arr[:, line, :] = zero_value
#         elif direction == 'xline':
#             del_num = int(random.uniform(low, high) * xline)
#             del_cols = random.sample(range(xline), del_num)
#             for line in del_cols:
#                 arr[:, :, line] = zero_value
#         else:
#             raise NotImplementedError('Unsupported direction')
#
#     elif misstype == 'c':
#         start_col = random.randrange(0, inline - del_num)
#         if direction == 'inline':
#             arr[:, start_col:start_col + del_num, :] = zero_value
#         elif direction == 'xline':
#             arr[:, :, start_col:start_col + del_num] = zero_value
#         else:
#             raise NotImplementedError('Unsupported direction')
#
#     elif misstype == 'allrandom':
#         # arr = np.array(arr)
#         arr = arr.reshape(sample, inline * xline)
#         del_num = int(random.uniform(low, high) * (inline * xline))
#         del_cols = random.sample(range(inline * xline), del_num)
#
#         for d in del_cols:
#             arr[:, d] = zero_value
#         arr = arr.reshape(sample, inline, xline)
#
#     else:
#         raise NotImplementedError('Unknown misstype')
#
#     return arr

# def remove_traces_3d_batch(arr, low=0.1, high=0.3,
#                      misstype='r', zero_value = 0., direction = 'None'):
#     '''
#     第一个维度：sample  第二个维度：inline  第三个维度：xline
#     :param arr: 2d numpy array
#     :param low:
#     :param high:
#     :param misstype: 'r' for random, 'p' for pattern, 'c' for consecutive
#     :return:
#     '''
#     arr = np.copy(arr)
#     B, sample, inline, xline = arr.shape
#     if low >= high:
#         raise ValueError('low must be less than high')
#
#     if misstype == 'r':
#         if direction == 'inline':
#             del_num = int(random.uniform(low, high) * inline)
#             del_cols = random.sample(range(inline), del_num)
#             for line in del_cols:
#                     arr[:, :, line, :] = zero_value
#         elif direction == 'xline':
#             del_num = int(random.uniform(low, high) * xline)
#             del_cols = random.sample(range(xline), del_num)
#             for line in del_cols:
#                     arr[:, :, :, line] = zero_value
#         else:
#             raise NotImplementedError('Unsupported direction')
#
#     elif misstype == 'c':
#         pass
#     elif misstype == 'allrandom':
#         # arr = np.array(arr)
#         arr = arr.reshape(B, sample, inline * xline)
#         del_num = int(random.uniform(low, high) * (inline * xline))
#         del_cols = random.sample(range(inline * xline), del_num)
#
#         for d in del_cols:
#             arr[:, :, d] = zero_value
#         arr = arr.reshape(B, sample, inline, xline)
#
#     else:
#         raise NotImplementedError('Unknown misstype')
#
#     return arr

def normalize_abs_max_by_slice(arr):
    _, _, s = arr.shape

    for i in range(s):
        arr_max = np.max(np.abs(arr[:, :, i]))
        arr[:, :, i] = arr[:, :, i] / arr_max

    return arr,


def denormalize_abs_max_by_slice(arr):
    _, _, s = arr.shape

    for i in range(s):
        arr_max = np.max(np.abs(arr[:, :, i]))
        arr[:, :, i] = arr[:, :, i] / arr_max

    return arr



def normalize_min_max(arr, arr_min = None, arr_max = None):
    """
        归一化数组，使其最大值为1，最小值为0.

        :param arr: 要归一化的数组
        :return: 归一化后的数组
    """
    if arr_min is None or arr_max is None:
        arr_max = np.max(arr)
        arr_min = np.min(arr)
    # 归一化
    normalized_arr = (arr - arr_min) / (arr_max - arr_min + 1e-6)
    return normalized_arr, arr_min, arr_max

def denormalize_min_max(arr, arr_min, arr_max):
    return arr * (arr_max - arr_min) + arr_min

def norm_by_log_max(arr):
    arr_norm = np.sign(arr) * np.log1p(np.abs(arr))
    arr_norm_max = np.max(np.abs(arr_norm))
    arr_norm = arr_norm / arr_norm_max
    return arr_norm, arr_norm_max


def denorm_by_log_max(arr_norm, arr_norm_max):
    arr_norm = arr_norm * arr_norm_max
    arr = np.sign(arr_norm) * np.expm1(np.abs(arr_norm))
    return arr

def norm_by_log(arr):
    arr_norm = np.sign(arr) * np.log1p(np.abs(arr))
    return arr_norm

def denorm_by_log(arr_norm):
    arr = np.sign(arr_norm) * np.expm1(np.abs(arr_norm))
    return arr

def denorm_by_log_tensor(arr_norm):
    arr = torch.sign(arr_norm) * torch.expm1(torch.abs(arr_norm))
    return arr

def norm_by_arcsinh_std(x, mean = None, std = None, eps=1e-6):
    x = np.arcsinh(x)


    if mean == None and std == None:
        mean = np.mean(x)
        std = np.std(x)

    x = (x - mean) / (std + eps)
    return x, mean, std

def denorm_by_arcsinh_std(x, mean, std):
    x = x * std + mean
    x = np.sinh(x)
    return x

def norm_by_arcsinh(x):
    return np.arcsinh(x)

def denorm_by_arcsinh(x):
    return np.sinh(x)

def calc_pad(length, window, stride):
    if length < window:
        return window - length
    return (stride - (length - window) % stride) % stride


def Ar2Patch(arr, win0, win1, stride0, stride1, padding=True):
    '''
    Cut an 2D-array into patches
    :param arr: 2D-array
    :param win1:
    :param win0:
    :param stride1:
    :param stride0:
    :return: network3D-array, the 1st dimension acts as a list
    '''
    rows, cols = arr.shape
    # print("shape of arr: ", arr.shape)
    # print("stride: ", stride_x, stride_y)
    # 计算需要添加的填充值
    if padding:
        pad0 = (stride0 - (rows - win0) % stride0) % stride0
        pad1 = (stride1 - (cols - win1) % stride1) % stride1
        # print("to be padded", pad_y, pad_x)
        # 添加填充
        arr_padded = np.pad(arr, ((0, pad0), (0, pad1)), mode='constant', constant_values=0)
        # print("padded shape: ", arr_padded.shape)
        # 提取小patch
        patches = []
        padded_rows, padded_cols = arr_padded.shape
        for row in range(0, padded_rows - win0 + 1, stride0):
            for col in range(0, padded_cols - win1 + 1, stride1):
                patch = arr_padded[row:row + win0, col:col + win1]
                patches.append(patch)
        return np.array(patches)
    else:
        patches = []
        for row in range(0, rows - win0 + 1, stride0):
            for col in range(0, cols - win1 + 1, stride1):
                patch = arr[row:row + win0, col:col + win1]
                patches.append(patch)
        return np.array(patches)


def Ar2Patch_3d(arr, win0, win1, win2, stride0, stride1, stride2, remove_zero):
    '''
    Cut an 3D-array into patches
    :param arr: 3D-array
    :param win0:
    :param win1:
    :param stride0:
    :param stride1:
    :return: network3D-array, the 1st dimension acts as a list
    '''
    sample, inline, xline = arr.shape
    # print("shape of arr: ", arr.shape)
    # print("stride: ", stride_x, stride_y)
    # 计算需要添加的填充值
    pad0 = calc_pad(inline, win0, stride0)
    pad1 = calc_pad(sample, win1, stride1)
    pad2 = calc_pad(xline, win2, stride2)
    # print("to be padded", pad_y, pad_x, pad_z)
    # 添加填充
    arr_padded = np.pad(arr, ((0, pad0), (0, pad1), (0, pad2)), mode='reflect')
    # arr_padded = np.pad(arr, ((0, pad0), (0, pad1), (0, pad2)), mode='constant', constant_values=0)
    print("padded shape: ", arr_padded.shape)
    # 提取小patch
    patches = []
    padded0, padded1, padded2 = arr_padded.shape
    for s in range(0, padded0 - win0 + 1, stride0):
        for i in range(0, padded1 - win1 + 1, stride1):
            for x in range(0, padded2 - win2 + 1, stride2):
                patch = arr_padded[s:s + win0, i:i + win1, x:x + win2]
                # print(patch.shape)
                if remove_zero:
                    # 去除全零patch
                    nonzero_ratio = np.count_nonzero(patch) / float(patch.size)
                    if nonzero_ratio < 0.6:
                        continue
                patches.append(patch)
    return patches

def Ar2Patch_5d(arr, wins, strides):
    '''
    Cut an 2D-array into patches
    :param arr: 2D-array
    :param win0:cd
    :param win1:
    :param stride0:
    :param stride1:
    :return: 5D-array, the 1st dimension acts as a list
    '''
    D0, D1, D2, D3, D4 = arr.shape
    S0, S1, S2, S3, S4 = strides
    W0, W1, W2, W3, W4 = wins
    # print("shape of arr: ", arr.shape)
    # print("stride: ", stride_x, stride_y)
    # 计算需要添加的填充值
    pad0 = (S0 - (D0 - W0) % S0) % S0
    pad1 = (S1 - (D1 - W1) % S1) % S1
    pad2 = (S2 - (D2 - W2) % S2) % S2
    pad3 = (S3 - (D3 - W3) % S3) % S3
    pad4 = (S4 - (D4 - W4) % S4) % S4
    # print("to be padded", pad_y, pad_x, pad_z)
    # 添加填充
    arr_padded = np.pad(arr, ((0, pad0), (0, pad1), (0, pad2), (0, pad3), (0, pad4)),
                        mode='constant', constant_values=0)
    # print("padded shape: ", arr_padded.shape)
    # 提取小patch
    patches = []
    padded0, padded1, padded2, padded3, padded4 = arr_padded.shape
    for d0 in range (0, padded0 - W0 + 1, S0):
        for d1 in range (0, padded1 - W1 + 1, S1):
            for d2 in range (0, padded2 - W2 + 1, S2):
                for d3 in range (0, padded3 - W3 + 1, S3):
                    for d4 in range (0, padded4 - W4 + 1, S4):
                        patch = arr_padded[d0:d0 + W0, d1:d1 + W1, d2:d2 + W2, d3:d3 + W3, d4:d4 + W4]
                        # print(patch.shape)
                        patches.append(patch)

    return np.array(patches)


def Combine(arr_list, win, rows, cols):
    '''
    Combine the patch in arr_list into a single array
    :param arr_list: a list of 2D-arrays (patch)
    :param win:
    :param rows:
    :param cols:
    :return: combined array
    '''
    result = np.zeros((rows, cols))
    pos_r = 0
    pos_c = 0
    for arr in arr_list:
        result[pos_r:pos_r + win, pos_c:pos_c + win] = arr
        pos_c += win
        if pos_c >= cols:
            pos_c = 0
            pos_r += win
    return result

def Combine_3d(arr_list, block_shape, samples, inline, xline):
    '''
    Combine the patch in arr_list into a single array
    :param arr_list: a list of 2D-arrays (patch)
    :param win:
    :param samples:
    :param inline:
    :return: combined array
    '''
    win0, win1, win2 = block_shape
    result = np.zeros((samples, inline, xline))
    pos0, pos1, pos2 = 0, 0, 0
    for arr in arr_list:
        result[pos0:pos0 + win0, pos1:pos1 + win1, pos2:pos2 + win2] = arr

        pos2 += win2
        if pos2 >= xline:
            pos2 = 0
            pos1 += win1

        if pos1 >= inline:
            pos1 = 0
            pos0 += win0
    return result

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def init_weights_norm(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)


def str2bool(v):
    '''
    Used when you input **bool** parameters in command line,
    in case the parser can't recognize bool values.
    :param v:
    :return:
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def pad(arr, d0, d1):
    num_rows, num_cols = arr.shape
    pad_rows = (d0 - (num_rows % d0)) % d0
    pad_cols = (d1 - (num_cols % d1)) % d1

    # Perform the padding
    padded_arr = np.pad(arr, ((0, pad_rows), (0, pad_cols))
                        , mode='constant', constant_values=0)

    return padded_arr

def pad_3d_reflect(arr, win0, win1, win2):

    sample, xline, inline = arr.shape

    pad0 = calc_pad(sample, win0, win0 // 2)
    pad1 = calc_pad(xline, win1, win1 // 2)
    pad2 = calc_pad(inline, win2, win2 // 2)


    # Perform the padding
    padded_arr = np.pad(arr, ((0, pad0), (0, pad1), (0, pad2))
                        , mode='reflect')

    return padded_arr

def pad_3d_constant(arr, win0, win1, win2):

    sample, xline, inline = arr.shape

    pad0 = calc_pad(sample, win0, win0 // 2)
    pad1 = calc_pad(xline, win1, win1 // 2)
    pad2 = calc_pad(inline, win2, win2 // 2)


    # Perform the padding
    padded_arr = np.pad(arr, ((0, pad0), (0, pad1), (0, pad2))
                        , mode='constant', constant_values=0)

    return padded_arr

def pad_5d(arr, divident1, divident2, divident3, divident4, divident5):
    D1, D2, D3, D4, D5 = arr.shape
    p1 = (divident1 - (D1 % divident1)) % divident1
    p2 = (divident2 - (D2 % divident2)) % divident2
    p3 = (divident3 - (D3 % divident3)) % divident3
    p4 = (divident4 - (D4 % divident4)) % divident4
    p5 = (divident5 - (D5 % divident5)) % divident5

    # Perform the padding
    padded_arr = np.pad(arr,
                        ((0, p1), (0, p2), (0, p3), (0, p4), (0, p5))
                        , mode='constant', constant_values=0)

    return padded_arr

# def h3(arr, win, model, device):
#     '''
#     half-window padding, half-window stride, half-window replace
#     :param arr:
#     :param win:
#     :param isx:
#     :return:
#     '''
#     # print(arr.shape)
#     if arr.shape[0] % (win // 2) != 0 or arr.shape[1] % (win // 2) != 0:
#         raise ValueError('rows/cols of arr can\'t be divided by (win/2)!')
#
#     patches = Ar2Patch(arr, win, win, win, win)
#     # print("11111", len(patches))
#     reconstructed_patches = []
#     predict_time = []
#     # 预测
#     for patch in patches:
#
#         patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#         start_time = time.time()
#         reconstructed_patch = model(patch_t)
#         end_time = time.time()
#         predict_time.append(end_time - start_time)
#         reconstructed_patch = reconstructed_patch.cpu().detach().numpy().squeeze()
#
#         reconstructed_patches.append(reconstructed_patch)
#     reconstructed_patches = np.array(reconstructed_patches)
#     # print(reconstructed_patches.shape)
#
#     print("Predicting time: ", sum(predict_time))
#     print("Number of parameters: ", num_parameters(model))
#
#     n_rows = arr.shape[0] // win
#     n_cols = arr.shape[1] // win
#     # print(n_rows, n_cols)
#     reconstructed_patches = reconstructed_patches.reshape(n_rows, n_cols, win, win)
#     # print(reconstructed_patches.shape)
#
#     # half-window padding
#     reconstructed_c_patches = []
#     padded_arr = np.pad(arr, ((0, 0), (win//2, win//2)), mode='constant', constant_values=0)
#     # half-window stride patching for cols
#     complement_patches = Ar2Patch(padded_arr, win, win, win, win)
#     for i, c_patch in enumerate(complement_patches):
#
#         c_patch_t = torch.tensor(c_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#         reconstructed_c_patch = model(c_patch_t).cpu().detach().numpy().squeeze()
#
#         reconstructed_c_patches.append(reconstructed_c_patch)
#     reconstructed_c_patches = np.array(reconstructed_c_patches)
#     reconstructed_c_patches = reconstructed_c_patches.reshape(n_rows, n_cols + 1, win, win)
#     print(reconstructed_c_patches.shape)
#     for i in range(reconstructed_c_patches.shape[0]):
#         for j in range(reconstructed_c_patches.shape[1]):
#             if j == 0:
#                 reconstructed_patches[i][j][:, 0:win // 4] = reconstructed_c_patches[i][j][:, win // 2:win // 2 + win // 4]
#             else:
#                 reconstructed_patches[i][j-1][:, win // 2 + win // 4:] = reconstructed_c_patches[i][j][:, win // 4:win // 2]
#                 if j != reconstructed_c_patches.shape[1] - 1:
#                     reconstructed_patches[i][j][:, 0:win // 4] = reconstructed_c_patches[i][j][:, win // 2:win // 2 + win // 4]
#
#
#     reconstructed_c_patches = []
#     padded_arr = np.pad(arr, ((win//2, win//2), (0, 0)), mode='linear_ramp', end_values=0)
#     # half-window stride patching
#     complement_patches = Ar2Patch(padded_arr, win, win, win, win)
#     for i, c_patch in enumerate(complement_patches):
#         c_patch_t = torch.tensor(c_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#         reconstructed_c_patch = model(c_patch_t).cpu().detach().numpy().squeeze()
#
#         reconstructed_c_patches.append(reconstructed_c_patch)
#     reconstructed_c_patches = np.array(reconstructed_c_patches)
#     reconstructed_c_patches = reconstructed_c_patches.reshape(n_rows + 1, n_cols, win, win)
#     print(reconstructed_c_patches.shape)
#     for j in range(reconstructed_c_patches.shape[1]):
#         for i in range(reconstructed_c_patches.shape[0]):
#             if i == 0:
#                 reconstructed_patches[i][j][0:win//4, :] = reconstructed_c_patches[i][j][win//2:win//2+win//4, :]
#             else:
#                 reconstructed_patches[i-1][j][win//2+win//4:, :] = reconstructed_c_patches[i][j][win//4:win//2, :]
#                 if i != reconstructed_c_patches.shape[0] - 1:
#                     reconstructed_patches[i][j][0:win//4, :] = reconstructed_c_patches[i][j][win//2:win//2+win//4, :]
#
#     return reconstructed_patches
#
#
# def h3_3d(arr, block_shape, model, device, patchnorm=False):
#     """
#     Three-dimensional extension of the half-window padding, half-window stride, half-window replace method.
#
#     :param arr: 3D numpy array with shape (采样点, inline, xline)
#     :param win: Window size for patches
#     :param model: PyTorch model for patch prediction
#     :param device: Device to run the model on (e.g., 'cuda' or 'cpu')
#     :param patchnorm: Whether to normalize patches before model prediction
#     :return: 3D reconstructed array with the same shape as `arr`
#     """
#
#     # Ensure that the first two dimensions are divisible by half window size
#     if arr.shape[0] % (block_shape[0] // 2) != 0 or arr.shape[1] % (block_shape[1] // 2) != 0:
#         raise ValueError("采样点和inline维度的大小应能被整除")
#
#     # Patch extraction along 采样点 and inline dimensions
#     patches = Ar2Patch_3d(arr, block_shape[0], block_shape[1], block_shape[2],
#                           block_shape[0], block_shape[1], block_shape[2])
#     recon_patches = []
#     predict_time = []
#     # 预测原始数据
#     # Model prediction for each patch
#     for patch in patches:
#
#         patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#         start_time = time.time()
#         reconstructed_patch = model(patch_t)
#         end_time = time.time()
#         predict_time.append(end_time - start_time)
#         reconstructed_patch = reconstructed_patch.cpu().detach().numpy().squeeze()
#
#         recon_patches.append(reconstructed_patch)
#     recon_patches = np.array(recon_patches)
#     # print(recon_patches.shape)
#
#     print("Predicting time: ", sum(predict_time))
#     print("Number of parameters: ", num_parameters(model))
#
#     n0 = arr.shape[0] // block_shape[0]
#     n1 = arr.shape[1] // block_shape[1]
#     n2 = 1
#
#     win0 = block_shape[0]
#     win1 = block_shape[1]
#     win2 = block_shape[2]
#
#     recon_patches = recon_patches.reshape(n0, n1, n2, win0, win1, win2)
#     # print(recon_patches.shape)
#
#     # Half-window padding for inline and 采样点 dimensions
#     # Process along the inline dimension
#     recon_c_patches = []
#     padded_arr = np.pad(arr, ((0, 0), (win1 // 2, win1 // 2), (0, 0)), mode='constant', constant_values=0)
#     c_patches = Ar2Patch_3d(padded_arr, win0, win1, win2, win0, win1, win2)
#
#     for i, c_patch in enumerate(c_patches):
#         c_patch_t = torch.tensor(c_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#         reconstructed_c_patch = model(c_patch_t).cpu().detach().numpy().squeeze()
#
#         recon_c_patches.append(reconstructed_c_patch)
#     recon_c_patches = np.array(recon_c_patches)
#     recon_c_patches = recon_c_patches.reshape(n0, n1 + 1, n2,
#                                                               win0, win1, win2)
#
#     # Replace half-window along inline dimension
#     for k in range(recon_c_patches.shape[2]):
#         for i in range(recon_c_patches.shape[0]):
#             for j in range(recon_c_patches.shape[1]):
#                 # print(1, recon_patches.shape, recon_c_patches.shape)
#
#                 if j == 0:
#                     recon_patches[i][j][k][:, 0:win1 // 4, :] = recon_c_patches[i][j][k][:,
#                                                                     win1 // 2:win1 // 2 + win1 // 4, :]
#                 else:
#                     recon_patches[i][j - 1][k][:, win1 // 2 + win1 // 4:, :] = recon_c_patches[i][j][k][:,
#                                                                                   win1 // 4:win1 // 2, :]
#                     if j != recon_c_patches.shape[1] - 1:
#                         recon_patches[i][j][k][:, 0:win1 // 4, :] = recon_c_patches[i][j][k][:,
#                                                                         win1 // 2:win1 // 2 + win1 // 4, :]
#
#     # Process along the 采样点 dimension
#     recon_c_patches = []
#     padded_arr = np.pad(arr, ((win0 // 2, win0 // 2), (0, 0), (0, 0)), mode='linear_ramp', end_values=0)
#     c_patches = Ar2Patch_3d(padded_arr, win0, win1, win2, win0, win1, win2)
#
#     for i, c_patch in enumerate(c_patches):
#         c_patch_t = torch.tensor(c_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#         reconstructed_c_patch = model(c_patch_t).cpu().detach().numpy().squeeze()
#
#         recon_c_patches.append(reconstructed_c_patch)
#     recon_c_patches = np.array(recon_c_patches)
#     recon_c_patches = recon_c_patches.reshape(n0 + 1, n1, n2,
#                                                               win0, win1, win2)
#
#     # Replace half-window along 采样点 dimension
#     for k in range(recon_c_patches.shape[2]):
#         for j in range(recon_c_patches.shape[1]):
#             for i in range(recon_c_patches.shape[0]):
#                 # print(2, recon_patches.shape, recon_c_patches.shape)
#                 if i == 0:
#                     recon_patches[i][j][k][0:win0 // 4, :, :]\
#                         = recon_c_patches[i][j][k][win0 // 2:win0 // 2 + win0 // 4, :, :]
#                 else:
#                     recon_patches[i - 1][j][k][win0 // 2 + win0 // 4:, :, :]\
#                         = recon_c_patches[i][j][k][win0 // 4:win0 // 2, :, :]
#                     if i != recon_c_patches.shape[0] - 1:
#                         recon_patches[i][j][k][0:win0 // 4, :, :] \
#                             = recon_c_patches[i][j][k][win0 // 2:win0 // 2 + win0 // 4, :, :]
#
#     return recon_patches


def test_with_sliding_window(arr,
                            patch_shape: tuple,
                            model,
                            device,
                            stride: tuple):
    # 定义步长为窗口大小的一半，实现滑动窗口分块
    step_r, step_c = stride

    # 获取数组的大小
    rows, cols = arr.shape
    reconstructed = np.zeros_like(arr)
    count_matrix = np.zeros_like(arr)  # 用来记录重叠区域的加权

    for i in range(0, rows - patch_shape[0] + 1, step_r):
        for j in range(0, cols - patch_shape[1] + 1, step_c):
            patch = arr[i:i + patch_shape[0], j:j + patch_shape[1]]

            patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                reconstructed_patch = model(patch_t).cpu().numpy().squeeze()


            # 将重建后的 patch 叠加到对应的位置
            reconstructed[i:i + patch_shape[0], j:j + patch_shape[1]] += reconstructed_patch
            count_matrix[i:i + patch_shape[0], j:j + patch_shape[1]] += 1

    # 处理重叠区域，取平均值
    reconstructed /= np.maximum(count_matrix, 1)

    return reconstructed


def test_with_sliding_window_3d(arr,
                                block_shape: tuple,
                                model,
                                device,
                                stride: tuple):

    step0, step1, step2 = stride

    # 获取三维数组的大小 (depth, height, width)
    shape0, shape1, shape2 = arr.shape

    # 初始化重建后的数组和计数矩阵
    reconstructed = np.zeros_like(arr)
    count_matrix = np.zeros_like(arr)  # 用来记录重叠区域的加权

    count = 0
    # 三维滑动窗口分块
    for d in range(0, shape0 - block_shape[0] + 1, step0):
        for i in range(0, shape1 - block_shape[1] + 1, step1):
            for j in range(0, shape2 - block_shape[2] + 1, step2):
                # 提取当前的三维 patch
                patch = arr[d:d + block_shape[0], i:i + block_shape[1], j:j + block_shape[2]]

                print(f'predicting {count}-th patch, '
                      f'[{d}:{d + block_shape[0]}, {i}:{i + block_shape[1]}, {j}:{j + block_shape[2]}]')
                count += 1
                # 转换为 PyTorch tensor，并加上 batch 和 channel 维度
                patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # 进行预测，不计算梯度
                with torch.no_grad():
                    reconstructed_patch = model(patch_t).cpu().numpy().squeeze()

                # 将重建后的 patch 叠加到对应的位置
                reconstructed[d:d + block_shape[0], i:i + block_shape[1], j:j + block_shape[2]] += reconstructed_patch
                count_matrix[d:d + block_shape[0], i:i + block_shape[1], j:j + block_shape[2]] += 1

    # 处理重叠区域，取平均值
    reconstructed /= np.maximum(count_matrix, 1)

    return reconstructed


def test_with_sliding_window_2d(arr,
                                patch_shape: tuple,
                                model,
                                device,
                                stride: tuple):

    step0, step1 = stride

    # 获取三维数组的大小 (depth, height, width)
    shape0, shape1 = arr.shape

    # 初始化重建后的数组和计数矩阵
    reconstructed = np.zeros_like(arr)
    count_matrix = np.zeros_like(arr)  # 用来记录重叠区域的加权

    count = 0
    # 三维滑动窗口分块
    for d in range(0, shape0 - patch_shape[0] + 1, step0):
        for i in range(0, shape1 - patch_shape[1] + 1, step1):
                # 提取当前的三维 patch
                patch = arr[d:d + patch_shape[0], i:i + patch_shape[1]]

                print(f'predicting {count}-th patch, '
                      f'[{d}:{d + patch_shape[0]}, {i}:{i + patch_shape[1]}]')
                count += 1
                # 转换为 PyTorch tensor，并加上 batch 和 channel 维度
                patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # 进行预测，不计算梯度
                with torch.no_grad():
                    reconstructed_patch = model(patch_t).cpu().numpy().squeeze()

                # 将重建后的 patch 叠加到对应的位置
                reconstructed[d:d + patch_shape[0], i:i + patch_shape[1]] += reconstructed_patch
                count_matrix[d:d + patch_shape[0], i:i + patch_shape[1]] += 1

    # 处理重叠区域，取平均值
    reconstructed /= np.maximum(count_matrix, 1)

    return reconstructed

def test_with_sliding_window_2d_from_3d(arr,
                                patch_shape: tuple,
                                model,
                                device,
                                stride: tuple):

    step0, step1, _ = stride

    # 获取三维数组的大小 (depth, height, width)
    shape0, shape1, shape2 = arr.shape

    # 初始化重建后的数组和计数矩阵
    reconstructed = np.zeros_like(arr)
    count_matrix = np.zeros_like(arr)  # 用来记录重叠区域的加权

    count = 0
    # 三维滑动窗口分块
    for d in range(0, shape0 - patch_shape[0] + 1, step0):
        for i in range(0, shape1 - patch_shape[1] + 1, step1):
                # 提取当前的三维 patch
                patch = arr[d:d + patch_shape[0], i:i + patch_shape[1], :]

                print(f'predicting {count}-th patch, '
                      f'[{d}:{d + patch_shape[0]}, {i}:{i + patch_shape[1]}, 1:{patch_shape[2]}]')
                count += 1
                # 转换为 PyTorch tensor，并加上 batch 和 channel 维度
                patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                B, C, H, W, X = patch_t.shape
                patch_t = patch_t.permute(0, 4, 1, 2, 3).contiguous().view(B * X, C, H, W)
                # 进行预测，不计算梯度
                with torch.no_grad():
                    reconstructed_patch = model(patch_t)
                    reconstructed_patch = reconstructed_patch.view(B, X, C, H, W).permute(0, 2, 3, 4, 1).contiguous()
                    reconstructed_patch = reconstructed_patch.cpu().numpy().squeeze()
                # 将重建后的 patch 叠加到对应的位置
                reconstructed[d:d + patch_shape[0], i:i + patch_shape[1], :] += reconstructed_patch
                count_matrix[d:d + patch_shape[0], i:i + patch_shape[1], :] += 1

    # 处理重叠区域，取平均值
    reconstructed /= np.maximum(count_matrix, 1)

    return reconstructed


def test_with_sliding_window_5d(arr,
                                patch_shape: tuple,
                                model,
                                device,
                                stride: tuple):
    """
    5D 数据滑动窗口重建
    arr: 输入 5D ndarray, shape = (d1, d2, d3, d4, d5)
    patch_shape: 窗口尺寸 (p1, p2, p3, p4, p5)
    stride: 滑动步长 (s1, s2, s3, s4, s5)
    """

    # 解包
    s1, s2, s3, s4, s5 = stride
    d1, d2, d3, d4, d5 = arr.shape
    p1, p2, p3, p4, p5 = patch_shape

    # 初始化输出
    reconstructed = np.zeros_like(arr)
    count_matrix = np.zeros_like(arr)

    count = 0
    # 5D 滑动窗口遍历
    for i1 in range(0, d1 - p1 + 1, s1):
        for i2 in range(0, d2 - p2 + 1, s2):
            for i3 in range(0, d3 - p3 + 1, s3):
                for i4 in range(0, d4 - p4 + 1, s4):
                    for i5 in range(0, d5 - p5 + 1, s5):

                        # 提取 patch
                        patch = arr[
                            i1:i1 + p1,
                            i2:i2 + p2,
                            i3:i3 + p3,
                            i4:i4 + p4,
                            i5:i5 + p5
                        ]

                        print(f'predicting {count}-th patch, '
                              f'[{i1}:{i1 + p1}, {i2}:{i2 + p2}, {i3}:{i3 + p3}, {i4}:{i4 + p4}, {i5}:{i5 + p5}]')
                        count += 1
                        # 转成 tensor，并加上Batch和Channel维度
                        patch_t = torch.tensor(patch, dtype=torch.float32)\
                                        .unsqueeze(0).unsqueeze(0).to(device)

                        # 推理
                        with torch.no_grad():
                            reconstructed_patch = model(patch_t).cpu().numpy().squeeze()

                        # 加入输出
                        reconstructed[
                            i1:i1 + p1,
                            i2:i2 + p2,
                            i3:i3 + p3,
                            i4:i4 + p4,
                            i5:i5 + p5
                        ] += reconstructed_patch

                        # 重叠次数统计
                        count_matrix[
                            i1:i1 + p1,
                            i2:i2 + p2,
                            i3:i3 + p3,
                            i4:i4 + p4,
                            i5:i5 + p5
                        ] += 1

    # 处理重叠区域，取平均
    reconstructed /= np.maximum(count_matrix, 1)

    return reconstructed

def load_data_auto(filepath, key=None):
    """
    自动识别 .mat 文件格式（v7.3 或旧版）并读取内容。
    :param filepath: .mat 文件路径
    :return: 读取后的数据字典
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    if key is None or key == '' or key == "":
        key = os.path.basename(filepath)
        key, _ = os.path.splitext(key)


    if filepath.lower().endswith('.mat'):
        try:
            # 尝试用 scipy 加载（适用于 v7.2 及以下）
            data = sio.loadmat(filepath)
            print(f"使用 scipy.io 读取成功（旧版 MAT 文件）: {filepath}")
            return data[key]
        except NotImplementedError as e:
            print(f"scipy.io 无法读取，可能是 v7.3 HDF5 文件，尝试使用 h5py：{e}")
        except Exception as e:
            print(f"scipy.io.loadmat 其他错误：{e}")
            raise e

    if filepath.lower().endswith('.mat') or \
        filepath.lower().endswith('.h5'):
        # 如果是 v7.3 格式，使用 h5py 读取
        try:
            with h5py.File(filepath, 'r') as f:
                print(f"使用 h5py 成功打开 HDF5 格式的 MAT 文件：{filepath}")
                key = '/' + key
                data = f[key][:]
                data = data.transpose()
                return data
        except Exception as e:
            print(f"h5py 读取失败：{e}")
            raise e

    if filepath.lower().endswith('.npy'):
        data = np.load(filepath)
        return data



#
# def setup_distributed_environment(selected_gpus):
#     '''
#     设置分布式训练环境
#     :param selected_gpus: 可用的 GPU 列表
#     '''
#     num_gpus = len(selected_gpus)
#     if num_gpus > 1:
#         # 选择 GPU
#         os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))
#         os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#
#         # 计算 world_size 和 rank
#         world_size = num_gpus
#         rank = int(os.environ.get('RANK', 0))  # 从环境变量获取 rank
#         local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 从环境变量获取 local_rank
#
#         os.environ['WORLD_SIZE'] = str(world_size)
#         os.environ['LOCAL_RANK'] = str(local_rank)
#         os.environ['RANK'] = str(rank)
#
#         # 初始化分布式环境
#         torch.distributed.init_process_group(backend='nccl', init_method='env://')
#         torch.cuda.set_device(local_rank)
#         print(f"Distributed environment setup complete. Local rank: {local_rank}, Global rank: {dist.get_rank()}")
#     else:
#         # 单 GPU 或不需要分布式设置
#         print("Single GPU mode or no distributed setup required.")
#         if torch.cuda.is_available():
#             os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))
#             os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#         else:
#             print("No CUDA devices available.")

# 打印模型参数数量
def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


# def custom_loss(output, target):
#     '''
#     MSE + L0-norm
#
#     '''
#     alpha = 0.5
#
#     loss = (alpha * torch.mean((output - target) ** 2) +
#             (1 - alpha) * torch.nonzero(x).size(0))
#
#     return loss


# 保存模型、优化器状态和轮数
def save_checkpoint(model, optimizer, scheduler, epoch,
                    timelist, results_train, results_val, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'results_train': results_train,
        'results_val': results_val,
        'timelist': timelist,
    }, file_path)

def load_checkpoint(model, optimizer, scheduler,
                     file_path = ""):
    checkpoint = torch.load(file_path, map_location="cpu")
    # checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    results_train = checkpoint['results_train']
    results_val = checkpoint['results_val']
    timelist = checkpoint['timelist']
    start_epoch = checkpoint['epoch'] + 1  # 从保存的下一个epoch开始
    return start_epoch, results_train, results_val, timelist


def refill(arr, reconstructed, mask):
    reconstructed = arr * mask + reconstructed * (1 - mask)
    difference = reconstructed - arr

    return reconstructed, difference


def add_gaussian_noise(data, snr_db=10):
    """
    为三维地震数据添加高斯白噪声

    Parameters
    ----------
    data : np.ndarray
        输入三维地震数据，shape = (625, 201, 201)
    snr_db : float
        目标信噪比（dB）

    Returns
    -------
    noisy_data : np.ndarray
        加噪后的数据
    noise : np.ndarray
        添加的高斯噪声
    """

    # 计算原始信号功率
    signal_power = np.mean(data ** 2)

    # 根据目标 SNR 计算噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10))

    # 生成高斯白噪声
    noise = np.random.normal(
        loc=0,
        scale=np.sqrt(noise_power),
        size=data.shape
    )

    # 添加噪声
    noisy_data = data + noise

    return noisy_data, noise


class net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


if __name__ == '__main__':

    # seed_dict = []
    # for i in range(1, 9):
    #     seed_dict.append(i * 666)
    #
    # arr = np.random.rand(322, 40, 8)
    # remove_traces_3d(arr, 0.5, 0.5, 'ur', ur_factor=4, seed_dict=seed_dict)
    # print(arr)

    # arr = np.random.rand(64, 64, 32)
    # device = ('cuda:6')
    # from networks.network3D.UNet_3d import UNet_3d
    # model = UNet_3d().to(device)
    # recon = h3_3d(arr, (32, 32, 32), model, device)
    # print(3, recon.shape)
    # recon = recon.reshape(-1, 32, 32, 32)
    # print(3.5, arr.shape)
    # recon = Combine_3d(arr_list=recon, block_shape=(32, 32, 32),
    #            samples=arr.shape[0], inline=arr.shape[1], xline=arr.shape[2])
    #
    # print(4, recon.shape)
    # arr = np.random.rand(64, 64, 64)
    # model = net()
    # device = torch.device("cuda:2")
    # h3_3d(arr, (32, 32, 32), model, device)

    # seed_dict = {}
    # for count in range(20):
    #     seed_dict.update({count: count + 300})
    #
    # arr = np.random.randn(6, 6, 6)
    # arr = remove_traces_3d(arr, 0.5, 0.5, 'ur', ur_factor=4, seed=seed_dict)
    # print(arr[:, :, 0])
    # print(arr[:, :, 1])
    # print(arr[:, :, 2])
    #
    # # patches = Ar2Patch(arr, win_x=3, win_y=3, stride_x=2, stride_y=2)
    # # print("Number of patches:", len(patches))
    # # for patch in patches:
    # #     print(patch)


    # 示例数据
    # samples, inline, xline = 3, 4, 5  # 三维数组的维度
    # arr = np.random.rand(samples, inline, xline)  # 创建一个随机三维数组
    # processed_data = remove_traces_3d(arr, 0.3, 0.9, 'allrandom')
    # print(processed_data)

    data = load_data_auto('./sdata/45shots/test_shot25/shot25.mat')
    noisy, _ = add_gaussian_noise(data, snr_db=3)
    print("Original SNR:", array_SNR(noisy, data))
    print(data.dtype)
    print(noisy.dtype)