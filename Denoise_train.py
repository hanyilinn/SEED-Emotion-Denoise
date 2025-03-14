import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

from Denoise_config import *
from models import *

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

noise_type = 'EOG_SEED'
model_name = 'ASTI-Net'

print(noise_type)
print(model_name)

def data_prep(batch_size):
    # EOG path
    path = '/data2/ylhan/SEED/denoise_data/'

    train_input = np.load(path + f'X_train.npy')
    train_output = np.load(path + f'y_train.npy')

    val_input = np.load(path + f'X_val.npy')
    val_output = np.load(path + f'y_val.npy')

    test_input = np.load(path + f'X_test.npy')
    test_output = np.load(path + f'y_test.npy')
    
    # train_input = train_input.reshape(-1, *train_input.shape[2:])
    # train_output = train_output.reshape(-1, *train_output.shape[2:])
    # val_input = val_input.reshape(-1, *val_input.shape[2:])
    # val_output = val_output.reshape(-1, *val_output.shape[2:])
    # test_input = test_input.reshape(-1, *test_input.shape[2:])
    # test_output = test_output.reshape(-1, *test_output.shape[2:])
    

    trainset = my_dataset_nolabel(train_input, train_output)
    valset = my_dataset_nolabel(val_input, val_output)
    testset = my_dataset_nolabel(test_input, test_output)

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=10, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def train(model, device, train_dataloader, val_dataloader, epochs, learning_rate):
    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0,
          'amsgrad': False}]
    )

    criterion_mse = nn.MSELoss()
    best_val_loss = 200.0

    for epoch in range(epochs):

        total_train_loss_per_epoch = 0
        average_train_loss_per_epoch = 0
        train_step_num = 0

        model.train()
        for batch_idx, data in enumerate(
                tqdm(train_dataloader, desc='Training Progress')):  # batch_idx 是一个变量，用于表示当前批次的索引或编号
            train_step_num += 1
            x, y = data
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()  # clear up the grads of optimized Variable

            eeg_re = model(x)

            loss = criterion_mse(eeg_re.squeeze(), y.squeeze())

            total_train_loss_per_epoch = total_train_loss_per_epoch + loss.item()
            loss.backward()
            optimizer.step()

        average_train_loss_per_epoch = total_train_loss_per_epoch / train_step_num
        print("epoch-{}/{} Train Loss: {}\n".format(epoch + 1, epochs, average_train_loss_per_epoch))

        val_step_num = 0
        total_val_loss_per_epoch = 0
        average_val_loss_per_epoch = 0

        sum_acc = 0
        sum_rrmse = 0

        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                val_step_num += 1

                x, y = data
                x = x.float().to(device)
                y = y.float().to(device)

                eeg_re = model(x)

                loss = criterion_mse(eeg_re.detach().squeeze(), y.detach().squeeze())

                total_val_loss_per_epoch = total_val_loss_per_epoch + loss.item()

                # re EEG metrics
                acc = cal_ACC_tensor(eeg_re.detach().squeeze(), y.detach().squeeze())
                sum_acc += acc
                rrmse = cal_RRMSE_tensor(eeg_re.detach(), y.detach())
                sum_rrmse += rrmse

            average_val_loss_per_epoch = total_val_loss_per_epoch / val_step_num

            acc = sum_acc.item() / val_step_num
            rrmse = sum_rrmse.item() / val_step_num

            print("[epoch %d/%d] [LOSS: %f] [ACC: %f] [RRMSE: %f]" % (
            epoch + 1, epochs, average_val_loss_per_epoch, acc, rrmse))

            # wandb.log({"LOSS": average_val_loss_per_epoch, "ACC": acc, "RRMSE": rrmse})

            if average_val_loss_per_epoch < best_val_loss:
                print('save model ~')
                torch.save(model.state_dict(), f'checkpoint/{noise_type}_{model_name}.pkl')
                best_val_loss = average_val_loss_per_epoch

    # wandb.finish()


def test(model, test_dataloader):
    model.eval()
    test_step_num = 0

    acc_re_list = []
    rrmse_re_list = []
    snr_re_list = []

    eeg_outputs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_dataloader, desc='Testing Progress')):
            test_step_num += 1

            x, y = data
            x = x.float().to(device)
            y = y.float().to(device)

            eeg_re = model(x)
            eeg_outputs.append(eeg_re.cpu().numpy())

            # Calculation of three performance metrics for directly reconstructed EEGs
            acc = cal_ACC_tensor(eeg_re.detach().squeeze(), y.detach().squeeze()).item()
            acc_re_list.append(acc)

            rrmse = cal_RRMSE_tensor(eeg_re.detach().squeeze(), y.detach().squeeze()).item()
            rrmse_re_list.append(rrmse)

            snr = cal_SNR(eeg_re.squeeze(), y.squeeze())
            snr_re_list.append(snr)

    print('------------Denoising results for direct output------------------')

    acc_re_all = np.array(acc_re_list)
    rrmse_re_all = np.array(rrmse_re_list)
    acc_re = np.mean(acc_re_all)
    rrmse_re = np.mean(rrmse_re_all)
    snr_re_all = np.array(snr_re_list)
    snr_re = np.mean(snr_re_all)
  
    acc_mean = np.mean(acc_re_all)
    acc_std = np.std(acc_re_all)
    rrmse_mean = np.mean(rrmse_re_all)
    rrmse_std = np.std(rrmse_re_all)
    snr_mean = np.mean(snr_re_all)
    snr_std = np.std(snr_re_all)


    print(acc_re_all)
    print("acc_re = " + str(acc_re))
    print(rrmse_re_all)
    print("rrmse_re = " + str(rrmse_re))
    print(snr_re_all)
    print("snr_re = " + str(snr_re))
    
    print("ACC: Mean = {:.4f}, Std = {:.4f}".format(acc_mean, acc_std))
    print("RRMSE: Mean = {:.4f}, Std = {:.4f}".format(rrmse_mean, rrmse_std))
    print("SNR: Mean = {:.4f}, Std = {:.4f}".format(snr_mean, snr_std))


    all_eeg = np.concatenate(eeg_outputs, axis=0).squeeze()
    print(all_eeg.shape)

    return acc_re_list, rrmse_re_list, snr_re_list, all_eeg


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    seed = 1
    print(noise_type)
    setup_seed(seed)

    batch_size = 8
    epochs = 20
    learning_rate = 1e-4

    cha_num = 64
    eeg_length = 800

    train_dataloader, val_dataloader, test_dataloader = data_prep(batch_size)
        
    model = ASTI_Net(eeg_length, cha_num)
    model.apply(weights_init)
    model = model.to(device)

    device_ids = [3, 2]
    if torch.cuda.device_count() > 1:
        print(f"Using GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)

    # model.load_state_dict(torch.load(f'./checkpoint/{noise_type}_{model_name}.pkl'))
    train(model, device, train_dataloader, val_dataloader, epochs, learning_rate)

    print('----------------------------------------')
    del model

    model = ASTI_Net(eeg_length, cha_num)
    model.to(device)
    device_ids = [3, 2]
    if torch.cuda.device_count() > 1:
        print(f"Using GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(f'./checkpoint/{noise_type}_{model_name}.pkl'))

    acc, rrmse, snr, eeg_test_re = test(model, test_dataloader)
