import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched

from adapter import LLamaAdapter

def train_one_epoch(model: LLamaAdapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    ## model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    next_repairllama_cache=None
    for data_iter_step, (
            reapirllama_examples, repairllama_labels, codellama_examples, codellama_labels) in enumerate(
                metric_logger.log_every(data_loader, print_freq, header)
            ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
             codellama_loss, next_repairllama_cache = model(reapirllama_examples, codellama_examples,
                                                              repairllama_labels=repairllama_labels,
                                                              codellama_labels=codellama_labels,
                                                              next_repairllama_cache=next_repairllama_cache)
        codellama_loss_value = codellama_loss.item()
        if not math.isfinite(codellama_loss_value):
            print("Loss is {}, stopping training".format(codellama_loss_value))
            sys.exit(1)

        codellama_loss /= accum_iter
        loss_scaler(codellama_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=codellama_loss_value)
        # metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        codellama_loss_value_reduce = misc.all_reduce_mean(codellama_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('codellama_train_loss', codellama_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}