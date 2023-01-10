import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
from torch.nn import functional as F
from copy import deepcopy

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
from model.supernet_transformer import MaxLayer
import random
import time


def sample_configs(choices, dist=(0.333, 0.333, 0.333)):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads', 'mul']
    depth = random.choices(choices['depth'], dist)[0]
    for dimension in dimensions:
        config[dimension] = [random.choices(choices[dimension], dist)[0] for _ in range(depth)]

    config['embed_dim'] = [random.choices(choices['embed_dim'], dist)[0]]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,
                    know_distill=True, linear_eval=None, max_kd=None, random_kd=False, sandwich_training=1, 
                    arch_weight=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)
    optimizer2 = deepcopy(optimizer)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    print("Arch Weight:", arch_weight)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if know_distill:
            # sample random config
            if mode == 'super':
                config = sample_configs(choices=choices, dist=[0., 0., 1.])
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)
            elif mode == 'retrain':
                config = retrain_config
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            if amp:
                with torch.cuda.amp.autocast():
                    if teacher_model:
                        with torch.no_grad():
                            teach_output = teacher_model(samples)
                        _, teacher_label = teach_output.topk(1, 1, True, True)
                        outputs = model(samples)
                        outputs_from_super = outputs.detach()
                        loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                    else:
                        outputs = model(samples)
                        outputs_from_super = outputs.detach()
                        loss = criterion(outputs, targets)

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

            else:
                outputs = model(samples)
                outputs_from_super = outputs.detach()
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    loss = criterion(outputs, targets)

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()

            #########################
            # refer output recreate
            if False:
                with torch.no_grad():
                    if amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(samples)
                            outputs_from_super = outputs.detach()
                    else:
                        outputs = model(samples)
                        outputs_from_super = outputs.detach()
            #########################

            # train max kd between various results of models
            if max_kd is not None:
                optimizer.zero_grad()
                outputs_refer = outputs_from_super
                kd_list = []
                for sampling_dist in [[0.3, 0.3, 0.3], [0.1, 0.3, 0.6]]:
                    config = sample_configs(choices=choices, dist=sampling_dist)
                    model_module = unwrap_model(model)
                    model_module.set_sample_config(config=config)

                    if amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(samples)
                            loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer))#/2. + criterion(outputs, targets)/2.
                            kd_list.append(loss)
                    else:
                        outputs = model(samples)
                        loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer))#/2. + criterion(outputs, targets)/2.
                        kd_list.append(loss)

                kd_max_loss = max_kd(torch.stack(kd_list))
                if amp:
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    loss_scaler(kd_max_loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order, step_or_not=True)
                else:
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    kd_max_loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()

            # train random model
            if random_kd:
                '''
                if linear_eval is not None:
                    dist_temp = [[0.3, 0.3, 0.3]]
                else:
                    dist_temp = [[0.0, 0.3, 0.3], [0.3, 0.3, 0.0]]
                '''
                dist_temp = [[0.0, 0.3, 0.3], [0.3, 0.3, 0.0]]
                outputs_refer = outputs_from_super
                for sampling_dist in dist_temp:
                    config = sample_configs(choices=choices, dist=sampling_dist)
                    model_module = unwrap_model(model)
                    model_module.set_sample_config(config=config)

                    if amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(samples)
                            outputs_temp = outputs.detach()
                    else:
                        outputs = model(samples)
                        outputs_temp = outputs.detach()

                    optimizer.zero_grad()
                    if amp:
                        with torch.cuda.amp.autocast():
                            # loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer))# /2. + criterion(outputs, targets)/2.
                            # loss = kl_loss(F.log_softmax(outputs_refer), F.log_softmax(outputs))# Reverse KL
                            '''
                            a_sig = torch.nn.functional.softmax(arch_weight)
                            a_dis = a_sig / torch.max(a_sig)
                            a_sample = torch.bernoulli(a_dis)
                            rev_a = 1. - a_dis.detach()
                            ss = (a_dis + rev_a) * a_sample
                            loss = ss[0]*kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer)) \
                                    + ss[1]*criterion(outputs, targets)
                            '''
                            loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer)) \
                                    + criterion(outputs, targets)
                        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                        loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order, step_or_not=True)
                    else:
                        # loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer))# /2. + criterion(outputs, targets)/2.
                        # loss = kl_loss(F.log_softmax(outputs_refer), F.log_softmax(outputs))# Reverse KL
                        '''
                        a_sig = torch.nn.functional.softmax(arch_weight)
                        a_dis = a_sig / torch.max(a_sig)
                        a_sample = torch.bernoulli(a_dis)
                        rev_a = 1. - a_dis.detach()
                        ss = (a_dis + rev_a) * a_sample
                        loss = ss[0]*kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer)) \
                                + ss[1]*criterion(outputs, targets)
                        '''
                        loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer)) \
                                + criterion(outputs, targets)
                        loss.backward()
                        # arch_weight.grad *= -1.
                        optimizer.step()
                    outputs_refer = outputs_temp
                    torch.cuda.synchronize()

            '''sandwich training code back up
            optimizer.zero_grad()
            for sandwich_idx in range(sandwich_training):
                config = sample_configs(choices=choices, dist=(0.333, 0.333, 0.333))
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)

                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(samples)
                else:
                    outputs = model(samples)

                if amp:
                    with torch.cuda.amp.autocast():
                        loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_from_big))  # /2. + criterion(outputs, targets)/2.
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order, step_or_not=True if sandwich_idx==sandwich_training-1 else False)
                else:
                    loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_from_big))    # /2. + criterion(outputs, targets)/2.
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    loss.backward()
                    if sandwich_idx == sandwich_training-1:
                        optimizer.step()
            torch.cuda.synchronize()
            '''

            # train random model with linear evaluator
            if linear_eval is not None:
                config = sample_configs(choices=choices, dist=(0.333, 0.333, 0.333))
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)

                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(samples)
                        outputs_refer = linear_eval(outputs_from_super) ## linear_eval
                else:
                    outputs = model(samples)
                    outputs_refer = linear_eval(outputs_from_super) ## linear_eval

                optimizer.zero_grad()
                if amp:
                    with torch.cuda.amp.autocast():
                        loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer))  # /2. + criterion(outputs, targets)/2.
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order, step_or_not=True)
                else:
                    loss = kl_loss(F.log_softmax(outputs), F.log_softmax(outputs_refer))    # /2. + criterion(outputs, targets)/2.
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()

        else:
            # sample random config
            if mode == 'super':
                config = sample_configs(choices=choices, dist=(0.333, 0.333, 0.333))
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)
            elif mode == 'retrain':
                config = retrain_config
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            if amp:
                with torch.cuda.amp.autocast():
                    if teacher_model:
                        with torch.no_grad():
                            teach_output = teacher_model(samples)
                        _, teacher_label = teach_output.topk(1, 1, True, True)
                        outputs = model(samples)
                        loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                    else:
                        outputs = model(samples)
                        loss = criterion(outputs, targets)
            else:
                outputs = model(samples)
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            if amp:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    print(model.get_alphas())
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
