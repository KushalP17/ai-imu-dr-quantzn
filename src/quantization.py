import torch
import torch.nn as nn


def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max


def linear_quantize(
    fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8
) -> torch.Tensor:
    """
    linear quantization for single fp_tensor
      from
        fp_tensor = (quantized_tensor - zero_point) * scale
      we have,
        quantized_tensor = int(round(fp_tensor / scale)) + zero_point
    :param tensor: [torch.(cuda.)FloatTensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :param scale: [torch.(cuda.)FloatTensor] scaling factor
    :param zero_point: [torch.(cuda.)IntTensor] the desired centroid of tensor values
    :return:
        [torch.(cuda.)FloatTensor] quantized tensor whose values are integers
    """
    assert fp_tensor.dtype == torch.float
    assert isinstance(scale, float) or (
        scale.dtype == torch.float and scale.dim() == fp_tensor.dim()
    )
    assert isinstance(zero_point, int) or (
        zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()
    )

    scaled_tensor = fp_tensor / scale
    rounded_tensor = torch.round(scaled_tensor).to(dtype)
    shifted_tensor = rounded_tensor + zero_point

    quantized_min, quantized_max = get_quantized_range(bitwidth)
    return shifted_tensor.clamp_(quantized_min, quantized_max)


def get_quantization_scale_for_weight(weight, bitwidth):
    """
    get quantization scale for single tensor of weight
    :param weight: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [integer] quantization bit width
    :return:
        [floating scalar] scale
    """
    # we just assume values in weight are symmetric
    # we also always make zero_point 0 for weight
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    return fp_max / quantized_max


def linear_quantize_weight_per_channel(tensor, bitwidth):
    """
    linear quantization for weight tensor
        using different scales and zero_points for different output channels
    :param tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [torch.(cuda.)Tensor] scale tensor
        [int] zero point (which is always 0)
    """
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]

    scale = torch.zeros(num_output_channels, device=tensor.device)

    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        scale[oc] = get_quantization_scale_for_weight(_subtensor, bitwidth)
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)

    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0


def quantize_weights(layers, bitwidth=8):
    quantized_layers = [None] * len(layers)
    for i, layer in enumerate(layers):
        quantized_layers[i] = linear_quantize_weight_per_channel(
            layer.weight.data.to(torch.float), bitwidth
        )
    return quantized_layers


def record_activation_range(model, sample_function, sample_data):
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools

        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.ReLU)):
                all_hooks.append(
                    m.register_forward_hook(
                        functools.partial(_record_range, module_name=name)
                    )
                )
        return all_hooks

    hooks = add_range_recoder_hook(model)
    sample_function(sample_data)

    for h in hooks:
        h.remove()

    return input_activation, output_activation


class QuantizedConv1d(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        weight_scale,
        stride,
        padding,
        dilation,
        groups,
        weight_bitwidth=8,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("weight_scale", weight_scale)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        # TODO replace with int8 math, requires int8 bias
        # int8 math performed first with int8 weight and bias,
        # then floating point scale applied?
        weight_fp = self.weight.double() * self.weight_scale
        return nn.functional.conv1d(
            x,
            weight_fp,
            self.bias.double(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias, weight_scale):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("weight_scale", weight_scale)

    def forward(self, x):
        # TODO replace with int8 math
        weight_fp = self.weight.double() * self.weight_scale
        return nn.functional.linear(x, weight_fp, self.bias.double())
