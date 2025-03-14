from New_SEED_utils import build_preprocessed_eeg_dataset_CNN, RawEEGDataset, subject_independent_data_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from models import *


def denoise(input, denoise_model, apply_denoise):
    if apply_denoise:
        with torch.no_grad():
            denoised_input = denoise_model(input)  # 假设模型输出的形状与输入相同
        return denoised_input
    else:
        return input


# Train the model
def train():
    writer = SummaryWriter('/data2/ylhan/SEED/log')
    total_step = len(train_data_loader)
    batch_cnt = 0
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_data_loader):
            features = features.to(device) # torch.Size([24, 1, 62, 800])
            labels = labels.to(device) 
            
            denoised_features = denoise(features, denoise_model, apply_denoise)
            outputs = model(denoised_features)
            loss = criterion(outputs, labels)
            batch_cnt += 1
            writer.add_scalar('train_loss', loss, batch_cnt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                         total_step, loss.item()))
        scheduler.step()
        test_1()
    
    if apply_denoise == False:
        torch.save(model.state_dict(), f'/data2/ylhan/SEED/checkpoint/{model_name}.pth')
    else:
        torch.save(model.state_dict(), f'/data2/ylhan/SEED/checkpoint/{model_name}_ASTI_Net.pth')


# Test the model
def test_1(is_load=False):
    if is_load:
        if apply_denoise == False:
            model.load_state_dict(torch.load(f'/data2/ylhan/SEED/checkpoint/{model_name}.pth'))
        else:
            model.load_state_dict(torch.load(f'/data2/ylhan/SEED/checkpoint/{model_name}_ASTI_Net.pth'))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            denoised_features = denoise(features, denoise_model, apply_denoise)
            output = model(denoised_features)
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy is {}%'.format(100 * correct / total))


def test(is_load=False):
    if is_load:
        if apply_denoise == False:
            model.load_state_dict(torch.load(f'/data2/ylhan/SEED/checkpoint/{model_name}.pth'))
        else:
            model.load_state_dict(torch.load(f'/data2/ylhan/SEED/checkpoint/{model_name}_ASTI_Net.pth'))
        
    model.eval()
        
    all_labels = []
    all_preds = []
        
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            denoised_features = denoise(features, denoise_model, apply_denoise)
            output = model(denoised_features)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                
            # Collect all labels and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        print('Test Accuracy is {}%'.format(accuracy))
            
        precision = precision_score(all_labels, all_preds, average='weighted')  # Weighted average for multi-class
        recall = recall_score(all_labels, all_preds, average='weighted')  # Weighted average for multi-class
        f1 = f1_score(all_labels, all_preds, average='weighted')  # Weighted average for multi-class
        cm = confusion_matrix(all_labels, all_preds)  # Confusion Matrix

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{cm}')
        
        return accuracy, precision, recall, f1



if __name__ == "__main__":
    folder_path = '/data2/ylhan/SEED/SEED_EEG/Preprocessed_EEG/'
    feature_vector_dict, label_dict = build_preprocessed_eeg_dataset_CNN(folder_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10
    num_classes = 3
    batch_size = 24
    learning_rate = 0.0001
    
    denoise_model = ASTI_Net(800,62).to(device)
    state_dict = torch.load(f'/data2/ylhan/SEED/checkpoint/EOG_SEED_ASTI-Net.pkl')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    denoise_model.load_state_dict(state_dict)
    # apply_denoise = True
    apply_denoise = False
    
    model_name = 'TSception'
    # model_name = 'ShallowConvNet'
    # model = ConvNet(num_classes).to(device)
    

    all_subjects = [str(i) for i in range(1, 16)] # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    group_size = 3
    n_groups = len(all_subjects) // group_size 
    groups = [set(all_subjects[i * group_size:(i + 1) * group_size]) for i in range(n_groups)]
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for test_subject_set in groups:
        train_subjects = set(all_subjects) - test_subject_set
        print(f"Currently processing test subjects: {test_subject_set}")
        
        train_feature, train_label, test_feature, test_label = subject_independent_data_split(
            feature_vector_dict, label_dict, test_subject_set) # {'1'} {'2'}...

        desire_shape = [1, 62, 800]
        train_data = RawEEGDataset(train_feature, train_label, desire_shape)
        test_data = RawEEGDataset(test_feature, test_label, desire_shape)

        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

        input_size = (1, 62, 800)  # Input shape (1 channel, 62 channels, 800 data points)
        sampling_rate = 200  # Assuming the sampling rate is 256 Hz (adjust as needed)
        num_T = 32  # Example number of temporal filters (adjust based on your experiment)
        num_S = 64  # Example number of spatial filters (adjust based on your experiment)
        hidden = 256  # Example number of hidden units in the fully connected layer
        dropout_rate = 0.5  # Dropout rate for regularization
        model = TSception(num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate).to(device)
        
        # model = shallowConvNet(nChan=62, nTime=800).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

        train()
        accuracy, precision, recall, f1 = test(is_load=True)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    avg_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f'Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}')
    print(f'Average Precision: {avg_precision:.4f} ± {std_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f} ± {std_recall:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}')
