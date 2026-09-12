import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import config
from utilities.checkpoint import last_and_best, save
from utilities.tensorboard_myutils import add_scalars
import utilities.interface as interface
import utilities.metrics as metrics
import utilities.dataset_split as dataset_split
import utilities.dataset_get as dataset_get

from models.ConvNeXt import ConvNeXt
from models.EfficientNetV2 import EfficientNetV2
from models.ResNeXt import ResNeXt


def main():
    print("START")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


    # ----------------- DATA -----------------
    dataset = dataset_get.eurosat_rgb()

    # ----------------- MODEL -----------------
    model = ConvNeXt(num_classes=config.CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)


    # ----------------- LOADERS & CHECKPOINT PREP -----------------
    ckpt_path, best_ckpt_path, scaler_path, best_valid_avg_loss, start_epoch, model, optimizer = last_and_best(
        model, optimizer, device
        )

    train_loader, val_loader, test_loader = dataset_split.subset(dataset, scaler_path)

    writer = SummaryWriter(f"runs/{model.nameid}/exp_{config.EXPERIMENT_NAME}") # tensorboard --logdir=runs
    sample_x, _ = next(iter(train_loader))
    writer.add_graph(model, sample_x.to(device).float())


    # ----------------- THE LOOP -----------------
    for epoch in range(start_epoch, config.EPOCHS):

        train_result = interface.run_epoch(model, device, train_loader, criterion, optimizer)
        valid_result = interface.run_epoch(model, device, val_loader, criterion)

        valid_avg_loss = valid_result["avg_loss"]
        train_avg_loss = train_result["avg_loss"]

        #save
        #train_metrics = metrics.classification_metrics(train_result)
        valid_metrics = metrics.classification_metrics(valid_result)
        add_scalars(writer, valid_metrics, valid_avg_loss, train_avg_loss, epoch)
        save(ckpt_path, model, epoch, train_avg_loss, valid_avg_loss, optimizer, scaler_path)
        
        if valid_avg_loss < best_valid_avg_loss:
            best_valid_avg_loss = valid_avg_loss
            save(best_ckpt_path, model, epoch, train_avg_loss, valid_avg_loss, optimizer, scaler_path)

        #log
        writer.add_scalar("LOSS/TRAIN", train_avg_loss, epoch)
        writer.add_scalar("LOSS/VAL", valid_avg_loss, epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_train_loss: {train_avg_loss:.6f} | avg_val_loss: {valid_avg_loss:.6f}")
    
    writer.close()


    # ----------------- EVAL -----------------
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_result = interface.run_epoch(model, device, test_loader, criterion)
    test_metrics = metrics.classification_metrics(test_result)

    metrics.print_metrics(test_result, test_metrics, checkpoint)

    


if __name__ == "__main__":
    main()

