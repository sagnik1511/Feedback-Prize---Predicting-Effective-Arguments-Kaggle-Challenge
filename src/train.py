import yaml
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.bert_model import FeedbackBERT
from src.utils import calculate_scores, create_loaders
import warnings
warnings.filterwarnings("ignore")

softmax = nn.Softmax(dim=1)
train_step, val_step = 1, 1


def train_single_epoch(loader, model, optim, loss_fn):
    global train_step
    total_loss = 0.0
    all_targets = []
    all_preds = []
    for batch in tqdm(loader):
        optim.zero_grad()
        # Data Unpacking
        input_ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        token_type_ids = batch["token_type_ids"].cuda()
        targets = batch["target"].cuda()

        # Prediction & Loss Calculation
        pred = model(input_ids, mask, token_type_ids)
        loss = loss_fn(pred, targets)
        pred = torch.argmax(softmax(pred), dim=1)
        targets = targets.cpu().detach().numpy().tolist()
        pred = pred.cpu().detach().numpy().tolist()

        # Loss appending
        all_targets += targets
        all_preds += pred
        total_loss += loss.item()
        batch_scores = calculate_scores(targets, pred)

        # logging
        wandb.log(batch_scores, step=train_step)
        wandb.log({"training_loss" : loss.item()})
        train_step += 1
        loss.backward()
        optim.step()
        epoch_scores = calculate_scores(all_targets, all_preds)
        
    return model, optim, epoch_scores

def validate_single_epoch(loader, model, loss_fn):
    global val_step
    total_loss = 0.0
    all_targets = []
    all_preds = []
    for batch in tqdm(loader):
        # Data Unpacking
        input_ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        token_type_ids = batch["token_type_ids"].cuda()
        targets = batch["target"].cuda()

        # Prediction & Loss Calculation
        pred = model(input_ids, mask, token_type_ids)
        loss = loss_fn(pred, targets)
        pred = torch.argmax(softmax(pred), dim=1)
        targets = targets.cpu().detach().numpy().tolist()
        pred = pred.cpu().detach().numpy().tolist()

        # Loss appending
        all_targets += targets
        all_preds += pred
        total_loss += loss.item()
        batch_scores = calculate_scores(targets, pred, "validation")

        # logging
        wandb.log(batch_scores, step=val_step)
        wandb.log({"validation_loss" : loss.item()})
        val_step += 1
        epoch_scores = calculate_scores(all_targets, all_preds, "valdation")
    return total_loss, epoch_scores

def train(params):
    print("Process Initiated...")

    # Initializing objects

    print("Initializing DataLoaders...")
    train_loader, val_loader = create_loaders(params["dataset"])
    print("DataLoaders Generated...")
    print("Initializing Model...")
    model = FeedbackBERT(**params["model"]).cuda()
    print("Model Generated...")
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    if params["optimizer"]["name"] == "adam":
        optimizer = optim.Adam(model.parameters(),**params["optimizer"]["hparams"])
    else:
        raise NotImplementedError

    best_loss = torch.inf
    num_epochs = params["training"]["max_epochs"]
    early_stop_counter = 0
    checkpoint_path = params["training"]["chkp_path"]
    wandb.watch(model, log_freq=10)

    best_train_scores = None
    best_val_scores = None

    # Initializing training iterations
    for epoch in range(num_epochs):
        if early_stop_counter == params["training"]["early_stop_count"]:
            print("Iteration Stopper due to Repeatative Degradation...")
            break
        else:
            print(f"Epoch {epoch + 1}: ")
            model.train()
            model, optimizer, train_scores = train_single_epoch(train_loader, model,
                                                optimizer, loss_fn)
            print(train_scores)
            model.eval()
            current_loss, val_scores = validate_single_epoch(val_loader, model, loss_fn)
            print(val_scores)
            if current_loss < best_loss:
                best_loss = current_loss
                early_stop_counter = 0
                best_train_scores = train_scores
                best_val_scores = val_scores
                chkp = {"model_state": model.state_dict(),
                        "epoch": epoch+1}
                torch.save(chkp, checkpoint_path)
                print("Model Weights Updated...")
            else:
                early_stop_counter += 1
            print("\n")
    print("Process Finished...")
    print(f"Best Training Scores : {best_train_scores}")
    print(f"Best Validation Scores : {best_val_scores}")


if __name__ == '__main__':
    config_path = "config/__base__.yaml"
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
        f.close()
    print(params)
    wandb.init(config=params)
    train_step, val_step = 1, 1
    train(params)
    