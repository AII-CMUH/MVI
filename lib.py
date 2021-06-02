import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

import warnings
with warnings.catch_warnings():
    # Suppress TensorBoard warnings.
    warnings.filterwarnings('ignore', category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

softmax = nn.Softmax(dim=1)

class ImageFolderWithCf(torchvision.datasets.ImageFolder):
    def __init__(self, clinical_file_path, *args, **kwargs):
        super(ImageFolderWithCf, self).__init__(*args, **kwargs)
        self._clinical_df = pd.read_csv(clinical_file_path,
                                        index_col='filename',
                                        encoding='utf-8-sig')

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithCf, self).__getitem__(index)
        filename = os.path.basename(self.imgs[index][0])
        record = self._clinical_df.loc[filename]
        cf = record['age_normalized':'afp_normalized'].values.astype(np.float32)
        pid = int(record['pid'])
        return (original_tuple + (cf, pid))

def _convert_to_results_by_patients(pids, labels, pred_scores):
    def _loose_merge_binary_cls(classes):
        return int(np.any(classes))

    pid2result_by_img = {}
    for i in range(len(pids)):
        if pids[i] not in pid2result_by_img:
            pid2result_by_img[pids[i]] = {'labels': [], 'pred_scores': []}
        pid2result_by_img[pids[i]]['labels'].append(labels[i])
        pid2result_by_img[pids[i]]['pred_scores'].append(pred_scores[i])

    for data in pid2result_by_img.values():
        data['pred_scores'] = np.stack(data['pred_scores'], axis=0)

    new_labels = []
    new_pred_labels = []
    new_pred_scores = []
    for data in pid2result_by_img.values():
        label = _loose_merge_binary_cls(data['labels'])
        new_labels.append(label)

        pred_labels = np.argmax(data['pred_scores'], axis=1)
        pred_label = _loose_merge_binary_cls(pred_labels)
        new_pred_labels.append(pred_label)

        idx = np.argmax(data['pred_scores'][:,1])
        new_pred_scores.append(data['pred_scores'][idx,:])

    return (np.array(new_labels, dtype=labels.dtype),
            np.array(new_pred_labels, dtype=labels.dtype),
            np.stack(new_pred_scores, axis=0))

def _select_best_roc_th(fprs, tprs, ths, return_index=False):
    """
    Select the threshold that maximizes the sum of TPR and (1-FPR).
    """
    assert len(fprs) == len(tprs) and len(fprs) == len(ths), \
        '%d %d %d' % (len(fprs), len(tprs), len(ths))
    spe_tpr_sum = (1 - fprs) + tprs
    max_sum_indices = np.where(spe_tpr_sum == spe_tpr_sum.max())[0]

    abs_diff = np.abs(
        np.diff([1 - fprs[max_sum_indices], tprs[max_sum_indices]], axis=0)
    ).reshape(-1)
    min_diff_indices = np.where(abs_diff == abs_diff.min())[0]
    idx = max_sum_indices[min_diff_indices[0]]
    return_values = [ths[idx]]
    if return_index:
        return_values.append(idx)
    return return_values if len(return_values) > 1 else return_values[0]

def _convert_scores_to_classes(scores, threshold, dtype=np.int64):
    classes = np.zeros(scores.shape, dtype=dtype)
    classes[scores >= threshold] = 1
    return classes

def _write_data_to_tb(tb_writer, step, data):
    for key, val in data.items():
        if isinstance(val, dict):
            for k, v in val.items():
                if k == 'support':
                    continue
                tb_writer.add_scalar(key + '/' + k, v, step)
        else:
            tb_writer.add_scalar(key, val, step)
    tb_writer.flush()

def _save_and_keep_one_model(model_dir_path, model, epoch, step):
    save_file_path = os.path.join(model_dir_path, 'checkpoint-%d.pth' % step)
    torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step
        },
        save_file_path)

    # Delete old checkpoint files.
    file_paths = glob.glob(os.path.join(model_dir_path, 'checkpoint-*.pth'))
    to_remove_file_paths = [x for x in file_paths if x != save_file_path]
    for file_path in to_remove_file_paths:
        os.remove(file_path)

def train_eval(model_dir_path,
               dataloaders,
               model,
               num_epochs=None,
               eval_only=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(model_dir_path, exist_ok=True)
    if eval_only:
        num_epochs = 1
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.0003,
                                    momentum=0.9,
                                    weight_decay=0.00001)
        train_tbwriter = SummaryWriter(
            log_dir=os.path.join(model_dir_path, 'tensorboard_train'))
    val_tbwriter = SummaryWriter(
        log_dir=os.path.join(model_dir_path, 'tensorboard_val'))

    epoch = 0
    step = 0
    while num_epochs is None or epoch < num_epochs:
        epoch += 1
        for phase in ['train', 'val'] if not eval_only else ['val']:
            if phase == 'train':
                model.train()
            else:
                print('Running %s evaluation.' % phase)
                model.eval()

            batched_label_list = []
            batched_pid_list = []
            batched_pred_score_list = []
            running_loss = 0.0

            for inputs, labels, cfs, pids in dataloaders[phase]:
                batched_label_list.append(labels.numpy())
                batched_pid_list.append(pids.numpy())
                inputs = inputs.to(device)
                labels = labels.to(device)
                cfs = cfs.to(device)
                if phase == 'train':
                    optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs, cfs)
                    loss = criterion(logits, labels)
                    loss_val = loss.item()
                    if phase == 'train':
                        step += 1
                        loss.backward()
                        optimizer.step()
                        print('Epoch %d - Step %d - loss: %f'
                              % (epoch, step, loss_val))
                running_loss += loss_val * inputs.size(0)
                scores = softmax(logits)
                batched_pred_score_list.append(scores.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase])
            labels = np.concatenate(batched_label_list, axis=0)
            pids = np.concatenate(batched_pid_list, axis=0)
            pred_scores = np.concatenate(batched_pred_score_list, axis=0)
            labels, predicts, pred_scores = \
                _convert_to_results_by_patients(pids, labels, pred_scores)
            fprs, tprs, ths = roc_curve(labels, pred_scores[:,1])
            th = _select_best_roc_th(fprs, tprs, ths)
            predicts = _convert_scores_to_classes(
                pred_scores[:,1], th, dtype=predicts.dtype)
            auc = roc_auc_score(labels, pred_scores[:,1])
            report_dict = classification_report(
                labels, predicts, output_dict=True, zero_division=0)
            tb_data = {'Loss': epoch_loss, 'AUC': auc, 'score_th': th}
            tb_data.update(report_dict)

            if phase == 'train':
                _write_data_to_tb(train_tbwriter, step, tb_data)
                print('Epoch %d finished. Saving model to disk.' % epoch)
                _save_and_keep_one_model(model_dir_path, model, epoch, step)
            else:
                _write_data_to_tb(val_tbwriter, step, tb_data)
    if not eval_only:
        train_tbwriter.close()
    val_tbwriter.close()
