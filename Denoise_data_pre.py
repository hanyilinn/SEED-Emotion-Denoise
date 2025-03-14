import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.signal import resample

# 数据路径
clean_path = '/data2/ylhan/SEED/01_CleanEEG'
contaminated_path = '/data2/ylhan/SEED/02_ContaminatedEEG'

def down_signal(signal, original_fs, target_fs):
    num_samples = int(signal.shape[1] * target_fs / original_fs)
    down_sig = resample(signal, num_samples, axis=1)
    
    return down_sig


# 加载干净脑电和污染脑电数据
def load_data(clean_path, contaminated_path):
    clean_files = sorted(os.listdir(clean_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按文件名中的数字排序
    contaminated_files = sorted(os.listdir(contaminated_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按文件名中的数字排序

    clean_data = []
    contaminated_data = []
    original_fs = 250
    target_fs = 200
    
    for clean_file, contaminated_file in tqdm(zip(clean_files, contaminated_files), total=len(clean_files), desc="Processing files", unit="file"):
        # 加载文件
        clean_signal = np.load(os.path.join(clean_path, clean_file))
        contaminated_signal = np.load(os.path.join(contaminated_path, contaminated_file))
        
        clean_signal = down_signal(clean_signal, original_fs, target_fs)
        contaminated_signal = down_signal(contaminated_signal, original_fs, target_fs)        
        
        # 按照给定的规则进行裁剪：前4秒（1000个点），5到8秒（1000个点）
        clean_clips = [clean_signal[:, 0:800], clean_signal[:, 800:1600]]  # 第一段和第二段
        contaminated_clips = [contaminated_signal[:, 0:800], contaminated_signal[:, 800:1600]]  # 第一段和第二段
        
        # 将每段裁剪的数据添加到列表
        clean_data.extend(clean_clips)
        contaminated_data.extend(contaminated_clips)

    # 转换为NumPy数组
    clean_data = np.array(clean_data)
    contaminated_data = np.array(contaminated_data)
    
    return contaminated_data, clean_data

# 数据分割
def split_data(contaminated_data, clean_data):
    # 使用train_test_split进行数据分割，8:1:1比例
    X_train, X_temp, y_train, y_temp = train_test_split(contaminated_data, clean_data, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 保存数据
def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    # 设置保存路径
    save_path = '/data2/ylhan/SEED/denoise_data/'
    
    # 检查路径是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 保存数据
    np.save(os.path.join(save_path, 'X_train.npy'), X_train)
    np.save(os.path.join(save_path, 'X_val.npy'), X_val)
    np.save(os.path.join(save_path, 'X_test.npy'), X_test)
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(save_path, 'y_val.npy'), y_val)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)
    
    # 输出保存数据的形状
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

def main():
    # 加载干净和污染数据
    contaminated_data, clean_data = load_data(clean_path, contaminated_path)

    # 数据分割
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(contaminated_data, clean_data)

    # 保存数据
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == '__main__':
    main()
