# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.   

# base stuff
import os

# torch
import torch
import torch.distributed as dist

# custom stuff
from utils import metric


def validate(pargs, comm_rank, comm_size,
             device, step, epoch, 
             net, criterion, validation_loader, 
             logger, history_logger=None):
    
    logger.log_start(key = "eval_start", metadata = {'epoch_num': epoch+1})
    if pargs.history_logging:
        history_logger.on_validation_begin(epoch + 1)

    #eval
    net.eval()

    count_sum_val = torch.zeros((1), dtype=torch.float32, device=device)
    loss_sum_val = torch.zeros((1), dtype=torch.float32, device=device)
    iou_sum_val = torch.zeros((1), dtype=torch.float32, device=device)

    # disable gradients
    with torch.no_grad():

        # only print once per eval at most
        for inputs_val, label_val, filename_val in validation_loader:

            #send to device
            inputs_val = inputs_val.to(device, non_blocking=True)
            label_val = label_val.to(device, non_blocking=True)
            
            # forward pass
            outputs_val = net.forward(inputs_val)
            loss_val = criterion(outputs_val, label_val, reduce_mean=False)

            # accumulate loss
            loss_sum_val += loss_val
        
            #increase counter
            count_sum_val += len(label_val)
        
            # Compute score
            predictions_val = torch.argmax(torch.softmax(outputs_val, 1), 1)
            iou_val = metric.compute_score(predictions_val, label_val, num_classes=3, reduce_mean=False)
            iou_sum_val += iou_val
                
        # average the validation loss
        if dist.is_initialized():
            vals = torch.cat([count_sum_val, loss_sum_val, iou_sum_val])
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, async_op=False)
            count_sum_val, loss_sum_val, iou_sum_val = vals
        loss_avg_val = loss_sum_val.item() / count_sum_val.item()
        iou_avg_val = iou_sum_val.item() / count_sum_val.item()

    # print results
    logger.log_event(key = "eval_accuracy", value = iou_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.log_event(key = "eval_loss", value = loss_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})

    stop_training_target_reached = False
    if (iou_avg_val >= pargs.target_iou):
        logger.log_event(key = "target_accuracy_reached", value = pargs.target_iou, metadata = {'epoch_num': epoch+1, 'step_num': step})
        stop_training_target_reached = True

    # set to train
    net.train()

    logger.log_end(key = "eval_stop", metadata = {'epoch_num': epoch+1})
    if pargs.history_logging:
        history_logger.on_validation_end(epoch + 1, val_loss=loss_avg_val, val_accuracy=iou_avg_val)

    return stop_training_target_reached
