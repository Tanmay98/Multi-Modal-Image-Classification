from timm.models import create_model
# from timm.scheduler import create_scheduler_v2
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy
import time
import torch
from torch.nn import functional as F 
from dataloader import ImageDataset
import models2
import os
from tqdm import tqdm
import timm
from loss import byol_loss, ctrastive_loss, clip_loss
from torch.utils.tensorboard import SummaryWriter
from utils import plot_classes_preds, plot_input_images

device = "cuda:2" if torch.cuda.is_available() else "cpu"
train_dataset = ImageDataset(is_train=True)
val_dataset = ImageDataset(is_train=False)

print(len(train_dataset), len(val_dataset))

batch_size=50
# num_batches_train = int(len(train_dataset) / batch_size)

sampler_train = torch.utils.data.RandomSampler(train_dataset)
sampler_val = torch.utils.data.SequentialSampler(val_dataset)

data_loader_train = torch.utils.data.DataLoader(
    train_dataset, sampler=sampler_train,
    batch_size=batch_size,
)

data_loader_val = torch.utils.data.DataLoader(
    val_dataset, sampler=sampler_val,
    batch_size=batch_size
)

model = create_model('deit_small_patch16_224', num_classes=10)
model.to(device)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched='cosine', num_epochs=epochs)
criterion = torch.nn.CrossEntropyLoss()

output_dir = '/home/tbaweja/811/vit_results/checkpoints'
if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)

def early_stopping(curr_train_loss, prev_loss, min_delta, tolerance):
    counter = 0
    if curr_train_loss - prev_loss < min_delta:
        counter += 1
        if counter > tolerance:
            return True
        
print(f"Start training for {epochs} epochs")
start_time = time.time()
max_accuracy = 0.0
total_train_time = []
prev_loss = 0
loss_list = []
val_loss_list = []

writer = SummaryWriter("./runs/Exp3")

for epoch in tqdm(range(epochs)):
    torch.cuda.empty_cache()
    train_start_time = time.time()

    model.train(True)
    avg_loss = 0.0
    avg_ce_loss = 0.0
    avg_text_loss = 0.0
    lamb = 0.3

    for i,data in enumerate(data_loader_train):
        samples, targets, text_emb = data
        samples = samples.to(device)
        targets = targets.to(device)
        text_emb = text_emb.to(device)

        if i==0:
            writer.add_figure("Input Images", plot_input_images(samples), global_step=epoch)

        feat, outputs = model(samples)
        flat_out = torch.flatten(outputs)
        if True in torch.isnan(flat_out):
            print(f"Nan for batch {i}")
            
        ce_loss = criterion(outputs, targets)
        text_loss = ctrastive_loss(feat, text_emb)
        loss = ce_loss + lamb * text_loss    

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item()
        avg_text_loss += text_loss.item()
        avg_ce_loss += ce_loss.item()

        # plotting preds for 0 batch of for every epoch
        if i == 0:
            writer.add_figure("Predictions",
                              plot_classes_preds(model, samples, targets),
                              global_step=epoch * len(data_loader_train) + i)

    avg_loss = avg_loss / len(data_loader_train)
    avg_text_loss = avg_text_loss / len(data_loader_train)
    avg_ce_loss = avg_ce_loss / len(data_loader_train)
    
    writer.add_scalar("Train/Loss/Epochs", avg_loss, epoch)
    writer.add_scalar("Train/TextLoss/Epochs", avg_text_loss, epoch)
    writer.add_scalar("Train/CELoss/Epochs", avg_ce_loss, epoch)
 
    loss_list.append(f"For Epoch {epoch}: \ntotal loss: {avg_loss}\ttext loss: {avg_text_loss}\tce loss: {avg_ce_loss}")

    logfile = open("vit_results/TrainLogs.txt", "w")
    for res in loss_list:    
        logfile.write(f"{res}\n")
    logfile.close()

    total_train_time.append(time.time() - train_start_time)

    # lr_scheduler.step(epoch)  

    stop = early_stopping(avg_loss, prev_loss, min_delta=0.01, tolerance=10)
    if stop:
        break
    prev_loss = avg_loss
    
    if (epoch+1) % 10 == 0:
        checkpoint_path = os.path.join(output_dir, f'checkpoint{epoch}.pth')
    
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'lr_Scheduler': lr_scheduler.state_dict(),
        }, checkpoint_path)


    model.eval()
    
    with torch.no_grad():
        avg_val_loss = 0.0
        avg_top1 = 0
        avg_top5 = 0

        for i, data in enumerate(data_loader_val):
            samples, targets, _ = data
            samples = samples.to(device)
            targets = targets.to(device)
            
            _, output = model(samples)
            val_loss = criterion(output, targets)
            avg_val_loss += val_loss.item()

            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            avg_top1 += acc1.item()
            avg_top5 += acc5.item()

        avg_val_loss = avg_val_loss / len(data_loader_val)
        avg_top1, avg_top5 = avg_top1 / len(data_loader_val), avg_top5 / len(data_loader_val)
        val_loss_list.append(f"For Epoch {epoch}: \ntotal loss: {avg_val_loss}\tAvg Top-1: {avg_top1}\tAvg Top-5: {avg_top5}")

        logfile2 = open("vit_results/ValidationLogs.txt", "w")
        for res in val_loss_list:    
            logfile2.write(f"{res}\n")
        logfile2.close()

        writer.add_scalar("Validation/Loss/Epochs", avg_val_loss, epoch)
        writer.add_scalar("Validation/Top-1/Epochs", avg_top1, epoch)
        writer.add_scalar("Validation/Top-5/Epochs", avg_top5, epoch)

print("Total Train Time to run 50 epocs: ", sum(total_train_time)/ len(total_train_time))