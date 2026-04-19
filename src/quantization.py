import torch
import torch.nn as nn

DEFAULT_BITWIDTH = 8


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
    assert isinstance(scale, float) or (
        scale.dtype == torch.float and scale.dim() == fp_tensor.dim()
    )
    assert isinstance(zero_point, int) or (
        zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()
    )

    fp_tensor.to(torch.float)

    scaled_tensor = fp_tensor / scale
    rounded_tensor = torch.round(scaled_tensor).to(dtype)
    shifted_tensor = rounded_tensor + zero_point

    quantized_min, quantized_max = get_quantized_range(bitwidth)
    return shifted_tensor.clamp_(quantized_min, quantized_max)


def get_quantization_scale_and_zero_point(fp_tensor, bitwidth=DEFAULT_BITWIDTH):
    """
    get quantization scale for single tensor
    :param fp_tensor: [torch.(cuda.)Tensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [float] scale
        [int] zero_point
    """
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    zero_point = quantized_min - fp_min / scale

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else:  # convert from float to int using round()
        zero_point = round(zero_point)
    return scale, int(zero_point)


def get_quantization_scale_for_weight(weight, bitwidth=DEFAULT_BITWIDTH):
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


def linear_quantize_weight_per_channel(tensor, bitwidth=DEFAULT_BITWIDTH):
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


def linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale):
    """
    linear quantization for single bias tensor
        quantized_bias = fp_bias / bias_scale
    :param bias: [torch.FloatTensor] bias weight to be quantized
    :param weight_scale: [float or torch.FloatTensor] weight scale tensor
    :param input_scale: [float] input scale
    :return:
        [torch.IntTensor] quantized bias tensor
    """
    assert bias.dim() == 1
    assert isinstance(input_scale, float)

    bias.to(torch.float)
    if isinstance(weight_scale, torch.Tensor):
        assert weight_scale.dtype == torch.float
        weight_scale = weight_scale.view(-1)
        assert bias.numel() == weight_scale.numel()

    bias_scale = input_scale * weight_scale

    quantized_bias = linear_quantize(
        bias, 32, bias_scale, zero_point=0, dtype=torch.int32
    )
    return quantized_bias, bias_scale, 0


def shift_quantized_conv1d_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    shift quantized bias to incorporate input_zero_point for nn.Conv1d
        shifted_quantized_bias = quantized_bias - Conv(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert quantized_bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    return (
        quantized_bias - quantized_weight.sum((1, 2)).to(torch.int32) * input_zero_point
    )


def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    shift quantized bias to incorporate input_zero_point for nn.Linear
        shifted_quantized_bias = quantized_bias - Linear(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert quantized_bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    return quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point


def quantized_conv1d(
    input,
    weight,
    bias,
    feature_bitwidth,
    weight_bitwidth,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
    stride,
    padding,
    dilation,
    groups,
):
    """
    quantized 1d convolution
    :param input: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param weight_bitwidth: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.(cuda.)CharTensor] quantized output feature
    """
    assert input.dtype == torch.int8
    assert weight.dtype == input.dtype
    assert bias is None or bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    assert isinstance(output_zero_point, int)
    assert isinstance(input_scale, float)
    assert isinstance(output_scale, float)
    assert weight_scale.dtype == torch.float

    # Step 1: calculate integer-based 1d convolution (8-bit multiplication with 32-bit accumulation)
    if "cpu" in input.device.type and all(d == 1 for d in dilation):
        # use 32-b MAC for simplicity
        output = torch.nn.functional.conv1d(
            input.to(torch.int32),
            weight.to(torch.int32),
            None,
            stride,
            padding,
            dilation,
            groups,
        )
    else:
        # current version pytorch does not yet support integer-based conv1d() on GPUs or conv1d() when dilation is greater than 1
        output = torch.nn.functional.conv1d(
            input.float(), weight.float(), None, stride, padding, dilation, groups
        )
        output = output.round().to(torch.int32)
    if bias is not None:
        output = output + bias.view(1, -1, 1)

    # Step 2: scale the output
    output = output.float() * (
        (input_scale * weight_scale.view(1, weight_scale.shape[0], 1)) / output_scale
    )

    # Step 3: shift output by output_zero_point
    output += output_zero_point

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


def quantized_linear(
    input,
    weight,
    bias,
    feature_bitwidth,
    weight_bitwidth,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
):
    """
    quantized fully-connected layer
    :param input: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param weight_bitwidth: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.CharIntTensor] quantized output feature (torch.int8)
    """
    assert input.dtype == torch.int8
    assert weight.dtype == input.dtype
    assert bias is None or bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    assert isinstance(output_zero_point, int)
    assert isinstance(input_scale, float)
    assert isinstance(output_scale, float)
    assert weight_scale.dtype == torch.float

    # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
    if "cpu" in input.device.type:
        # use 32-b MAC for simplicity
        output = torch.nn.functional.linear(
            input.to(torch.int32), weight.to(torch.int32), bias
        )
    else:
        # current version pytorch does not yet support integer-based linear() on GPUs
        output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())

    # Step 2: scale the output
    output = output.float() * ((input_scale * weight_scale.view(1, -1)) / output_scale)

    # Step 3: shift output by output_zero_point
    output += output_zero_point

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


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
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        stride,
        padding,
        dilation,
        groups,
        pad_size=0,
        feature_bitwidth=DEFAULT_BITWIDTH,
        weight_bitwidth=DEFAULT_BITWIDTH,
        first_layer=False,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth
        self.first_layer = first_layer

        if pad_size > 0:
            self.pad = nn.ReplicationPad1d(pad_size)

    @classmethod
    def build(
        cls,
        conv_layer,
        conv_layer_name,
        relu_layer_name,
        input_activation,
        output_activation,
        pad_size,
        feature_bitwidth=DEFAULT_BITWIDTH,
        weight_bitwidth=DEFAULT_BITWIDTH,
        first_layer=False,
    ):
        input_scale, input_zero_point = get_quantization_scale_and_zero_point(
            input_activation[conv_layer_name], feature_bitwidth
        )
        output_scale, output_zero_point = get_quantization_scale_and_zero_point(
            output_activation[relu_layer_name], feature_bitwidth
        )
        quantized_weight, weight_scale, _ = linear_quantize_weight_per_channel(
            conv_layer.weight.data, weight_bitwidth
        )
        quantized_bias, _, _ = linear_quantize_bias_per_output_channel(
            conv_layer.bias.data, weight_scale, input_scale
        )
        shifted_quantized_bias = shift_quantized_conv1d_bias(
            quantized_bias, quantized_weight, input_zero_point
        )

        return cls(
            quantized_weight,
            shifted_quantized_bias,
            input_zero_point,
            output_zero_point,
            input_scale,
            weight_scale,
            output_scale,
            conv_layer.stride,
            conv_layer.padding,
            conv_layer.dilation,
            conv_layer.groups,
            pad_size=pad_size,
            feature_bitwidth=feature_bitwidth,
            weight_bitwidth=weight_bitwidth,
            first_layer=first_layer,
        )

    def forward(self, x):
        if self.first_layer:
            x = linear_quantize(
                x, self.feature_bitwidth, self.input_scale, self.input_zero_point
            )
        x = self.pad(x)
        return quantized_conv1d(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.weight_bitwidth,
            self.input_zero_point,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        feature_bitwidth=DEFAULT_BITWIDTH,
        weight_bitwidth=DEFAULT_BITWIDTH,
        first_layer=False,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth
        self.first_layer = first_layer

    @classmethod
    def build(
        cls,
        linear_layer,
        linear_layer_name,
        input_activation,
        output_activation,
        feature_bitwidth=DEFAULT_BITWIDTH,
        weight_bitwidth=DEFAULT_BITWIDTH,
        first_layer=False,
    ):
        input_scale, input_zero_point = get_quantization_scale_and_zero_point(
            input_activation[linear_layer_name], feature_bitwidth
        )
        output_scale, output_zero_point = get_quantization_scale_and_zero_point(
            output_activation[linear_layer_name], feature_bitwidth
        )
        quantized_weight, weight_scale, _ = linear_quantize_weight_per_channel(
            linear_layer.weight.data, weight_bitwidth
        )
        quantized_bias, _, _ = linear_quantize_bias_per_output_channel(
            linear_layer.bias.data, weight_scale, input_scale
        )
        shifted_quantized_bias = shift_quantized_linear_bias(
            quantized_bias, quantized_weight, input_zero_point
        )

        return cls(
            quantized_weight,
            shifted_quantized_bias,
            input_zero_point,
            output_zero_point,
            input_scale,
            weight_scale,
            output_scale,
            feature_bitwidth=feature_bitwidth,
            weight_bitwidth=weight_bitwidth,
            first_layer=first_layer,
        )

    def forward(self, x):
        if self.first_layer:
            x = linear_quantize(
                x, self.feature_bitwidth, self.input_scale, self.input_zero_point
            )
        return quantized_linear(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.weight_bitwidth,
            self.input_zero_point,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
        )


class DequantizeTanh(nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        x_float = (x.float() - self.zero_point) * self.scale
        return torch.tanh(x_float)
