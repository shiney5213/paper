import torch
import torch.nn.functional as F


def train_one_epoch(alexnet, optimizer, lr_scheduler, train_dataloader, device, epoch, total_steps, print_freq = 10):
    # lr upldate
    lr_scheduler.step()
    for imgs, classes in train_dataloader:
        imgs, classes = imgs.to(device), classes.to(device)
        
        # calculate the loss : cross entropy loss
        output = alexnet(imgs)
        loss = F.cross_entropy(output, classes)
        
        # update the parameters
        optimizer.zero_grad()
        loss.backword()
        optimizer.step()
        
        
        # log the information and add to tensorboad
        if total_steps % 10 == 0:
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes)
                loss = loss.item()
                accuracy = accuracy.item()
                
                print('Epoch: {}\tStep: {} \Loss: {:.4f} \tAcc: {}'.format(epoch + 1, total_steps, loss, accuracy))
                tbwriter.add_scalar('loss', loss, total_steps)
                tbwriter.add_scalar('accuracy', accuracy, total_steps)
                
        # print out gradient values and parameter average values
        if total_steps % 100 == 0:
            with torch.no_grad():
                # print and save the grad of the parameters
                # also print and save parameter values
                print('*' * 10)
                for name, parameter in alexnet.named_parameters():
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        print('\t{} - grad_avg: {}'.format(name, avg_grad))
                        tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                        tbwriter.add_histogram('grad/{}'.format(name), parameter.grad.cpu.numpy(), total_steps)
                    
                    if parameter.data in not None:
                        avg_weight = torch.mean(parameter.data)
                        print('\t{} - param_avg: {}'.format(name, avg_weight))
                        tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
                        tbwriter.add_histogram('weight/{}'.format(name), parameter.data.cpu().numpy(), total_steps)
                        
    total_steps += 1
    
    return total_steps, optimizer, alexnet, tbwriter
                        
                
                
                
                
        
        
        
        
    
    
    