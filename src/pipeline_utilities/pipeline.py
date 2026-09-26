import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import config
from pipeline_utilities.checkpoint import last_and_best, save
from pipeline_utilities.tensorboard_myutils import add_scalars
import pipeline_utilities.interface as interface
import pipeline_utilities.metrics as metrics
import pipeline_utilities.dataset_split as dataset_split

def ran(model:nn.Module, dataset, device:torch.device, criterion, ds_name):
    model.to(device)
    # ----------------- LOADERS & CHECKPOINT PREP -----------------
    ckpt_path, best_ckpt_path, scaler_path, best_valid_avg_loss, start_epoch, model, model.optimizer = last_and_best(
        model, model.optimizer, device, ds_name
        )

    train_loader, val_loader, test_loader = dataset_split.subset(dataset, scaler_path)

    writer = SummaryWriter(f"runs/{model.nameid}/exp_{config.EXPERIMENT_NAME}") # tensorboard --logdir=runs
    sample_x, _ = next(iter(train_loader))
    writer.add_graph(model, sample_x.to(device).float())


    # ----------------- THE LOOP -----------------
    for epoch in range(start_epoch, config.EPOCHS):

        train_result = interface.run_epoch(model, device, train_loader, criterion, model.optimizer)
        valid_result = interface.run_epoch(model, device, val_loader, criterion)

        valid_avg_loss = valid_result["avg_loss"]
        train_avg_loss = train_result["avg_loss"]

        #save
        #train_metrics = metrics.classification_metrics(train_result)
        valid_metrics = metrics.calculate(valid_result, dataset.classes, model.nameid)
        add_scalars(writer, valid_metrics, valid_avg_loss, train_avg_loss, epoch)
        save(ckpt_path, model, epoch, train_avg_loss, valid_avg_loss, model.optimizer, scaler_path)
        
        if valid_avg_loss < best_valid_avg_loss:
            best_valid_avg_loss = valid_avg_loss
            save(best_ckpt_path, model, epoch, train_avg_loss, valid_avg_loss, model.optimizer, scaler_path)

        #log
        writer.add_scalar("LOSS/TRAIN", train_avg_loss, epoch)
        writer.add_scalar("LOSS/VAL", valid_avg_loss, epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_train_loss: {train_avg_loss:.6f} | avg_val_loss: {valid_avg_loss:.6f}")
    
    writer.close()


    # ----------------- EVAL -----------------
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_result = interface.run_epoch(model, device, test_loader, criterion)
    test_metrics = metrics.calculate(test_result, dataset.classes, model.nameid)

    metrics.print_all(test_metrics, checkpoint)

    return test_metrics