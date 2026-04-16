import torch


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
    for layer in layers:
        quantized_weight, _, _ = linear_quantize_weight_per_channel(
            layer.weight.data.to(torch.float), bitwidth
        )
        # TODO write the quantized weights back into layers
