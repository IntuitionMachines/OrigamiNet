import torch
import warnings

from torch.utils.checkpoint import checkpoint


def checkpoint_sequential_step(functions, segments, *inputs, **kwargs):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        inputs: tuple of Tensors that are inputs to :attr:`functions`
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    # To accept variadic arguments is not consistent with nn.Sequential.
    # This interface will be changed at PyTorch 1.3.
    # See also: https://github.com/pytorch/pytorch/issues/19260
    if not inputs:
        warnings.warn('Giving no input to checkpoint_sequential has been deprecated, '
                      'a TypeError will be raised after PyTorch 1.3',
                      DeprecationWarning)
    elif len(inputs) > 1:
        warnings.warn('multiple inputs to checkpoint_sequential has been deprecated, '
                      'a TypeError will be raised after PyTorch 1.3',
                      DeprecationWarning)

    def run_function(start, end, functions):
        def forward(*inputs):
            for j in range(start, end + 1):
                if isinstance(inputs, tuple):
                    inputs = functions[j](*inputs)
                else:
                    inputs = functions[j](inputs)
            return inputs
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, len(functions)-segments, segments):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs,
                            preserve_rng_state=preserve)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

    return checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)
