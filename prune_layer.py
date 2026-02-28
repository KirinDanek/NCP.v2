"""
prune_layer.py

Purpose
-------
Utility functions for pruning convolutional filters in-place.

prune_conv_layer_sequential(...)
   - Performs structural pruning:
       - Rebuilds the target Conv2d with out_channels-1 and copies weights/bias except the
         pruned filter.
       - Finds the next Conv2d layer (skipping intervening ReLUs/Pool/etc.) and rebuilds it
         with in_channels-1 to match.
       - If pruning the last conv layer, it adjusts the first Linear layer in classifier
         to remove the corresponding input-channel block.
   - This changes model architecture and reduces compute.

Augmented vs vanilla support
----------------------------
get_conv_seq_and_name(model) abstracts over model layouts:
- Vanilla VGG16: model.features is the conv/pool sequential.
- AugmentedVGG16: conv/pool are split into model.before + encode + decode + model.after.
  The function flattens these into one sequential view for pruning.

After pruning, prune_conv_layer_sequential reassigns the pruned sequential back to:
- Augmented: model.before / model.encode / model.decode / model.after
- Vanilla:   model.features

Assumptions / risks
-------------------
- layer_index refers to index within the flattened sequential produced by get_conv_seq_and_name().
  For augmented models, indices include encode/decode if they exist.
- next Conv2d search assumes a sequential feed-forward topology (VGG-style).
- Bias is always created for new convs (bias=True), matching most VGG convs.
- Classifier adjustment assumes a standard VGG flatten → first Linear mapping where
  in_features is divisible by conv.out_channels.

Used by
-------
- prune_aug_vgg.py / prune_van_vgg.py during iterative pruning loops.
"""


import torch
from torch.nn.utils import prune as torch_prune
import torch.nn as nn
import numpy as np


def get_conv_seq_and_name(model):
    if hasattr(model, "features"): ### vanilla call
        return model.features, "features"
    elif hasattr(model, "before"): ### augmented call
        seq = []
        seq.extend(model.before)
        if hasattr(model, "encode"):
            seq.append(model.encode)
        if hasattr(model, "decode"):
            seq.append(model.decode)
        seq.extend(model.after)
        return torch.nn.Sequential(*seq), "augmented"
    else:
        raise ValueError("Unrecognized model structure")

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


def prune_conv_layer_sequential(model, layer_index, filter_index, cuda_flag=False):
    ''' input parameters
    1. model: 현재 모델
    2. layer_index: 자르고자 하는 layer index
    3. filter_index: 자르고자 하는 layer의 filter index
    '''
    # _, conv = model.features._modules.items()[layer_index]
    conv_seq, _ = get_conv_seq_and_name(model)
    _, conv = list(conv_seq._modules.items())[layer_index]  # 해당 layer를 우선 끄집어 온다.
    next_conv = None
    offset = 1

    while layer_index + offset < len(conv_seq._modules.items()):  # 전체 network의 layer 수보다 클때까지 while문 반복
        # res =  model.features._modules.items()[layer_index+offset]
        res = list(conv_seq._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):  # 현재 layer를 기준으로 다음 layer가 conv layer이냐?
            next_name, next_conv = res
            break
        offset = offset + 1

    # 새로운 conv_layer(new_conv)를 다시 생성시킨다.
    # output 쪽의 갯수를 하나 줄여준다.
    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels,
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=True)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    # for p in new_conv.parameters():
    #     p.requires_grad = False

    # weight는 해당 filter를 제외하고 총 갯수 - 1 를 넣어준다.
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

    # bias도 해당 filter number의 값을 제외하고 총 갯수 -1를 넣어준다.
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias).cuda() if cuda_flag else torch.from_numpy(bias)

    # 다음 conv layer도 받는 쪽의 layer 갯수를 줄여준다.
    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=True)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        # for p in next_new_conv.parameters():
        #     p.requires_grad = False

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
            *(replace_layers(conv_seq, i, [layer_index, layer_index + offset],
                             [new_conv, next_new_conv]) for i, _ in enumerate(conv_seq)))
        del conv_seq
        del conv

        conv_seq = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        conv_seq = torch.nn.Sequential(
            *(replace_layers(conv_seq, i, [layer_index], \
                             [new_conv]) for i, _ in enumerate(conv_seq)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(
            new_weights)

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

        # After replacing conv_seq
    if hasattr(model, "before"):
        total_len = len(model.before)
        model.before = nn.Sequential(*[conv_seq[i] for i in range(total_len)])
        if hasattr(model, "encode"):
            model.encode = conv_seq[total_len]
            model.decode = conv_seq[total_len + 1]
            total_len += 2
        model.after = nn.Sequential(*[conv_seq[i] for i in range(total_len, len(conv_seq))])
    else:
        model.features = conv_seq

    return model