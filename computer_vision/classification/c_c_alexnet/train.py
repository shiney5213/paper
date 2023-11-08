import numpy as np
import torch
import torch.nn.functional as F

from utils import acculate


# def train_one_epoch(alexnet, optimizer, lr_scheduler, train_dataloader, device, epoch, total_steps, print_freq = 10):
#     # lr upldate
#     lr_scheduler.step()
#     for imgs, classes in train_dataloader:
#         imgs, classes = imgs.to(device), classes.to(device)
        
#         # calculate the loss : cross entropy loss
#         output = alexnet(imgs)
#         loss = F.cross_entropy(output, classes)
        
#         # update the parameters
#         optimizer.zero_grad()
#         loss.backword()
#         optimizer.step()
        
        
#         # log the information and add to tensorboad
#         if total_steps % 10 == 0:
#             with torch.no_grad():
#                 _, preds = torch.max(output, 1)
#                 accuracy = torch.sum(preds == classes)
#                 loss = loss.item()
#                 accuracy = accuracy.item()
                
#                 print('Epoch: {}\tStep: {} \Loss: {:.4f} \tAcc: {}'.format(epoch + 1, total_steps, loss, accuracy))
#                 tbwriter.add_scalar('loss', loss, total_steps)
#                 tbwriter.add_scalar('accuracy', accuracy, total_steps)
                
#         # print out gradient values and parameter average values
#         if total_steps % 100 == 0:
#             with torch.no_grad():
#                 # print and save the grad of the parameters
#                 # also print and save parameter values
#                 print('*' * 10)
#                 for name, parameter in alexnet.named_parameters():
#                     if parameter.grad is not None:
#                         avg_grad = torch.mean(parameter.grad)
#                         print('\t{} - grad_avg: {}'.format(name, avg_grad))
#                         tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
#                         tbwriter.add_histogram('grad/{}'.format(name), parameter.grad.cpu.numpy(), total_steps)
                    
#                     if parameter.data in not None:
#                         avg_weight = torch.mean(parameter.data)
#                         print('\t{} - param_avg: {}'.format(name, avg_weight))
#                         tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
#                         tbwriter.add_histogram('weight/{}'.format(name), parameter.data.cpu().numpy(), total_steps)
                        
#     total_steps += 1
    
#     return total_steps, optimizer, alexnet, tbwriter


def train_loop(dataloader, model, loss_fn, optimizer, device, losses_df, NUM_BATCHES):
    
    output_list = []
    label_list = []
    train_loss = 0
    for i, batch in enumerate(dataloader):
        img = batch['image']
        label = batch['label']
        
        img, label = img.to(device, dtype = torch.float), label.to(device, dtype = torch.float)
        
        # foward propagation
        train_output = model(img)
        
        # calculate the loss : cross entropy loss
        # batch_loss = F.cross_entropy(train_output, labels)
        batch_loss = loss_fn(train_output, torch.max(label, 1)[1])
        
        
        # set gradient 0
        optimizer.zero_grad()
        # backpropagation
        batch_loss.backward()
        # update the parameters
        optimizer.step()
        
        train_output = train_output.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        output_list.extend(np.argmax(train_output, 1))
        label_list.extend(label)
        train_loss += batch_loss.item()
        
           
    # culuate loss
    train_loss = train_loss/NUM_BATCHES
    losses_df['train_loss'].append(train_loss)
    
    # culculate acc
    # print('lentrain_output', len(output_list))
    # print('train_output', output_list)
    # print('label', label)
          
    # print('labels', len(label_list))
    train_acc = acculate(label_list, output_list)
    losses_df['train_acc'].append(train_acc)
    
    print('train : loss - {:.4f}'.format(train_loss), 'acc - {:.4f}'.format(train_acc))
    
    return model, optimizer, losses_df

def val_roop(dataloader, model, loss_fn, optimizer, device, losses_df, NUM_BATCHES):
    
         
    val_loss = 0
    outputs_list = []
    labels_list = []
    # validation, test : not backpropagation
    with torch.no_grad():
        model.eval()
        
        
        for i , batch in enumerate(dataloader):
            imgs = batch['image']
            labels = batch['label']
        
            imgs, labels = imgs.to(device, dtype = torch.float), labels.to(device, dtype = torch.float)
            
            # forward propagation
            val_output = model(imgs)
            
            # get loss
            batch_loss = loss_fn(val_output, torch.max(labels, 1)[1])
            
            
            val_output = val_output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            outputs_list.extend(np.argmax(val_output, 1))
            labels_list.extend(labels)
            val_loss += batch_loss.item()
            
    # culuate loss
    val_loss = val_loss/NUM_BATCHES
    losses_df['val_loss'].append(val_loss)
    
    # culuate acc
    val_acc = acculate(labels_list, outputs_list)
    losses_df['val_acc'].append(val_acc)

    print('val : loss - {:.4f}'.format(val_loss), 'acc - {:.4f}'.format(val_acc))
    
    return losses_df, val_loss
    
    
    
        
    
                        
                
                
                
                
        
        
        
        
    
    
    