import types
import torch
import torch.nn as nn
from torch.autograd import Function


def CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add=False):
    x_size = tuple(x.size())

    s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)

    out_size = x_size[:-1] + (output_size,)

    # Broadcast s and compute x * s
    s = s.view(s_view)
    xs = x * s

    # Broadcast h then compute h:
    # out[h_i] += x_i * s_i
    h = h.view(s_view).expand(x_size)

    if force_cpu_scatter_add:
        out = x.new(*out_size).zero_().cpu()
        return out.scatter_add_(-1, h.cpu(), xs.cpu()).cuda()
    else:
        out = x.new(*out_size).zero_()
        return out.scatter_add_(-1, h, xs)


def CountSketchFn_backward(h, s, x_size, grad_output):
    s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)

    s = s.view(s_view)
    h = h.view(s_view).expand(x_size)

    grad_x = grad_output.gather(-1, h)
    grad_x = grad_x * s
    return grad_x


class CountSketchFn(Function):

    @staticmethod
    def forward(ctx, h, s, output_size, x, force_cpu_scatter_add=False):
        x_size = tuple(x.size())

        ctx.save_for_backward(h, s)
        ctx.x_size = tuple(x.size())

        return CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add)

    @staticmethod
    def backward(ctx, grad_output):
        h, s = ctx.saved_variables

        grad_x = CountSketchFn_backward(h, s, ctx.x_size, grad_output)
        return None, None, None, grad_x


class CountSketch(nn.Module):
    r"""Compute the count sketch over an input signal.

    .. math::

        out_j = \sum_{i : j = h_i} s_i x_i

    Args:
        input_size (int): Number of channels in the input array
        output_size (int): Number of channels in the output sketch
        h (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s (array, optional): Optional array of size input_size of -1 and 1.

    .. note::

        If h and s are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input: (...,input_size)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input_size, output_size, h=None, s=None):
        super(CountSketch, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        if h is None:
            h = torch.LongTensor(input_size).random_(0, output_size)
        if s is None:
            s = 2 * torch.Tensor(input_size).random_(0, 2) - 1

        # The Variable h being a list of indices,
        # If the type of this module is changed (e.g. float to double),
        # the variable h should remain a LongTensor
        # therefore we force float() and double() to be no-ops on the variable h.
        def identity(self):
            return self

        h.float = types.MethodType(identity, h)
        h.double = types.MethodType(identity, h)

        self.register_buffer('h', h)
        self.register_buffer('s', s)

    def forward(self, x):
        x_size = list(x.size())

        assert (x_size[-1] == self.input_size)

        return CountSketchFn.apply(self.h, self.s, self.output_size, x)


def ComplexMultiply_forward(X_re, X_im, Y_re, Y_im):
    Z_re = torch.addcmul(X_re * Y_re, X_im, Y_im, value= -1)
    Z_im = torch.addcmul(X_re * Y_im, X_im, Y_re, value= 1)
    return Z_re, Z_im


def ComplexMultiply_backward(x_re, x_im, y_re, y_im, grad_z_re, grad_z_im):
    grad_x_re = torch.addcmul(grad_z_re * y_re, grad_z_im, y_im, value= 1)
    grad_x_im = torch.addcmul(grad_z_im * y_re, grad_z_re, y_im, value= -1)
    grad_y_re = torch.addcmul(grad_z_re * x_re, grad_z_im, x_im, value= 1)
    grad_y_im = torch.addcmul(grad_z_im * x_re, grad_z_re, x_im, value= -1)
    return grad_x_re, grad_x_im, grad_y_re, grad_y_im


class ComplexMultiply(Function):
    @staticmethod
    def forward(ctx, x_re, x_im, y_re, y_im):
        ctx.save_for_backward(x_re, x_im, y_re, y_im)
        return ComplexMultiply_forward(x_re, x_im, y_re, y_im)

    @staticmethod
    def backward(ctx, grad_z_re, grad_z_im):
        x_re, x_im, y_re, y_im = ctx.saved_tensors
        return ComplexMultiply_backward(x_re, x_im, y_re, y_im, grad_z_re, grad_z_im)


class CompactBilinearPoolingFn(Function):
    @staticmethod
    def forward(ctx, h1, s1, h2, s2, output_size, x, y, force_cpu_scatter_add=False):
        ctx.save_for_backward(h1, s1, h2, s2, x, y)
        ctx.x_size = tuple(x.size())
        ctx.y_size = tuple(y.size())
        ctx.force_cpu_scatter_add = force_cpu_scatter_add
        ctx.output_size = output_size

        # Compute the count sketch of each input: x ==> px, y ==> py
        px = CountSketchFn_forward(h1, s1, output_size, x, force_cpu_scatter_add)
        fx = torch.fft.rfft(px, dim=1)
        fx_re = fx.real
        fx_im = fx.imag
        del px
        py = CountSketchFn_forward(h2, s2, output_size, y, force_cpu_scatter_add)
        fy = torch.fft.rfft(py, dim=1)
        fy_re = fy.real
        fy_im = fy.imag
        del py

        # Convolution of the two sketch using an FFT.
        # Compute the FFT of each sketch

        # Complex multiplication: element-wise product
        prod_re, prod_im = ComplexMultiply_forward(fx_re, fx_im, fy_re, fy_im)
        complex_prod = torch.complex(prod_re, prod_im)

        # Back to real domain
        # The imaginary part should be zero's
        re = torch.fft.irfft(complex_prod, n=output_size)

        return re

    @staticmethod
    def backward(ctx, grad_output):
        h1, s1, h2, s2, x, y = ctx.saved_tensors

        # Recompute part of the forward pass to get the input to the complex product
        # Compute the count sketch of each input
        px = CountSketchFn_forward(h1, s1, ctx.output_size, x, ctx.force_cpu_scatter_add)
        py = CountSketchFn_forward(h2, s2, ctx.output_size, y, ctx.force_cpu_scatter_add)

        # Then convert the output to Fourier domain
        grad_output = grad_output.contiguous()
        grad_prod = torch.fft.rfft(grad_output, dim=1)
        grad_re_prod = grad_prod.real
        grad_im_prod = grad_prod.imag

        # Compute the gradient of x first then y

        # Gradient of x
        # Recompute fy
        fy = torch.fft.rfft(py, dim=1)
        re_fy = fy.real
        im_fy = fy.imag
        del py
        # Compute the gradient of fx, then back to temporal space
        grad_re_fx = torch.addcmul(grad_re_prod * re_fy, grad_im_prod, im_fy, value= 1)
        grad_im_fx = torch.addcmul(grad_im_prod * re_fy, grad_re_prod, im_fy, value= -1)
        complex_fx = torch.complex(grad_re_fx, grad_im_fx)
        grad_fx = torch.fft.irfft(complex_fx, n=ctx.output_size)
        # Finally compute the gradient of x
        grad_x = CountSketchFn_backward(h1, s1, ctx.x_size, grad_fx)
        del re_fy, im_fy, grad_re_fx, grad_im_fx, grad_fx

        # Gradient of y
        # Recompute fx
        fx = torch.fft.rfft(px, dim=1)
        re_fx = fx.real
        im_fx = fx.imag
        del px
        # Compute the gradient of fy, then back to temporal space
        grad_re_fy = torch.addcmul(grad_re_prod * re_fx, grad_im_prod, im_fx, value= 1)
        grad_im_fy = torch.addcmul(grad_im_prod * re_fx, grad_re_prod, im_fx, value= -1)
        complex_fy = torch.complex(grad_re_fy, grad_im_fy)
        grad_fy = torch.fft.irfft(complex_fy, n=ctx.output_size)
        # Finally compute the gradient of y
        grad_y = CountSketchFn_backward(h2, s2, ctx.y_size, grad_fy)
        del re_fx, im_fx, grad_re_fy, grad_im_fy, grad_fy

        return None, None, None, None, None, grad_x, grad_y, None


class CompactBilinearPooling(nn.Module):
    r"""Compute the compact bilinear pooling between two input array x and y

    .. math::

        out = \Psi (x,h_1,s_1) \ast \Psi (y,h_2,s_2)

    Args:
        input_size1 (int): Number of channels in the first input array
        input_size2 (int): Number of channels in the second input array
        output_size (int): Number of channels in the output array
        h1 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s1 (array, optional): Optional array of size input_size of -1 and 1.
        h2 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s2 (array, optional): Optional array of size input_size of -1 and 1.
        force_cpu_scatter_add (boolean, optional): Force the scatter_add operation to run on CPU for testing purposes

    .. note::

        If h1, s1, s2, h2 are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input 1: (...,input_size1)
        - Input 2: (...,input_size2)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input1_size, input2_size, output_size, h1=None, s1=None, h2=None, s2=None,
                 force_cpu_scatter_add=False):
        super(CompactBilinearPooling, self).__init__()
        self.add_module('sketch1', CountSketch(input1_size, output_size, h1, s1))
        self.add_module('sketch2', CountSketch(input2_size, output_size, h2, s2))
        self.output_size = output_size
        self.force_cpu_scatter_add = force_cpu_scatter_add

    def forward(self, x, y=None):
        if y is None:
            y = x
        return CompactBilinearPoolingFn.apply(self.sketch1.h, self.sketch1.s, self.sketch2.h, self.sketch2.s,
                                              self.output_size, x, y, self.force_cpu_scatter_add)
