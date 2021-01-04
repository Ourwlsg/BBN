from sklearn.metrics import classification_report
from tqdm import tqdm

import main._init_paths
from lib.core.evaluate import accuracy, AverageMeter, FusionMatrix

import numpy as np
import torch
import time


def train_model(
        trainLoader,
        model,
        epoch,
        epoch_number,
        optimizer,
        combiner,
        l_format,
        criterion,
        cfg,
        logger,
        **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)
    combiner.reset_l(l_format)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        loss, now_acc = combiner.forward(model, criterion, image, label, meta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model(dataLoader, epoch_number, model, cfg, criterion, logger, device, target_names, **kwargs):
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)
    labels_epoch = []
    prediction_epoch = []
    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            labels_epoch.extend(label.numpy())
            image, label = image.to(device), label.to(device)

            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True)
            loss = criterion(output, label)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            prediction_epoch.extend(now_result.cpu().numpy())
            all_loss.update(loss.data.item(), label.shape[0])
            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)
        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc.avg * 100
        )
        logger.info(pbar_str)
    result_epoch = classification_report(labels_epoch, prediction_epoch, labels=[0, 1, 2, 3, 4],
                                         target_names=target_names,
                                         output_dict=True, digits=3)
    print(classification_report(labels_epoch, prediction_epoch, labels=[0, 1, 2, 3, 4], target_names=target_names,
                                digits=3))
    pbar.close()
    return acc.avg, all_loss.avg, result_epoch
