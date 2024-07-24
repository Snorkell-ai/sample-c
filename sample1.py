"""PyTorch tools"""
from collections.abc import Sequence, Callable, Iterator, Iterable, Generator, Mapping
from typing import Optional, Any, Literal
import logging
from contextlib import contextmanager
import functools
import random
import math
from types import EllipsisType
from contextlib import nullcontext
from itertools import zip_longest
import torch
import torch.utils.hooks, torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from .python_tools import type_str, try_copy, EndlessContinuingIterator, Compose, reduce_dim
CUDA_IF_AVAILABLE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def ensure_device(x, device:Optional[torch.device]):
    """Recursively moves an object to the specified device.

    This function checks the type of the input object `x` and moves it to
    the specified `device` if it is a PyTorch tensor. If `x` is a list or
    tuple, it recursively applies the same operation to each element. If
    `device` is None, it simply returns the input object without any
    modifications. Note that moving tensors to a device can be slow,
    especially for large objects or when done repeatedly.

    Args:
        x (Union[torch.Tensor, list, tuple]): The input object to be moved to the device.
        device (Optional[torch.device]): The target device to move the object to.

    Returns:
        Union[torch.Tensor, list, tuple]: The input object moved to the specified device,
        or the original object if no device is specified or if it is not a
            tensor.
    """
    if device is None: return x
    if isinstance(x, torch.Tensor): return x.to(device)
    elif isinstance(x, (list, tuple)): return [ensure_device(i, device) for i in x]
    else: return x

def ensure_detach(x) -> Any:
    """Recursively detaches a tensor or a collection of tensors.

    This function checks the type of the input `x`. If `x` is a PyTorch
    tensor, it detaches it from the current computation graph. If `x` is a
    list or tuple, it recursively detaches each element in the collection.
    For any other type, it simply returns the input unchanged. Note that
    detaching can be a slow operation when applied to large tensors or deep
    collections.

    Args:
        x (Any): The input which can be a tensor, list, tuple, or any other type.

    Returns:
        Any: The detached tensor, or a collection of detached tensors, or the input
            unchanged.
    """
    if isinstance(x, torch.Tensor): return x.detach()
    elif isinstance(x, (list, tuple)): return [ensure_detach(i) for i in x]
    else: return x

def ensure_cpu(x) -> Any:
    """Recursively moves a tensor or a collection of tensors to the CPU.

    This function checks if the input `x` is a PyTorch tensor. If it is, the
    tensor is moved to the CPU. If `x` is a list or tuple, the function
    recursively applies itself to each element in the collection, ensuring
    that all tensors within are moved to the CPU. If `x` is neither a tensor
    nor a collection, it is returned unchanged. Note that this operation can
    be slow, especially for large collections of tensors.

    Args:
        x (Any): The input data which can be a tensor, list, tuple, or any other type.

    Returns:
        Any: The input data with tensors moved to the CPU, or the input unchanged if
            it is not a tensor or collection.
    """
    if isinstance(x, torch.Tensor): return x.cpu()
    elif isinstance(x, (list, tuple)): return [ensure_cpu(i) for i in x]
    else: return x

def ensure_detach_cpu(x) -> Any:
    """Recursively detaches a tensor and moves it to the CPU, if possible.

    This function checks the type of the input `x`. If `x` is a PyTorch
    tensor, it detaches the tensor from its current computation graph and
    moves it to the CPU. If `x` is a list or tuple, it recursively applies
    the same operation to each element in the collection. For any other
    type, it simply returns the input unchanged. Note that this operation
    can be slow, especially for large tensors or deep nested structures.

    Args:
        x (Any): The input which can be a tensor, list, tuple, or any other type.

    Returns:
        Any: The detached tensor moved to the CPU, or the input unchanged if it is
            not a tensor or collection.
    """
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    elif isinstance(x, (list, tuple)): return [ensure_detach_cpu(i) for i in x]
    else: return x

def ensure_float(x) -> Any:
    """Convert input to float if possible.

    This function attempts to convert the input `x` to a float. If `x` is a
    single-element PyTorch tensor, it will convert it to a float after
    detaching it from the computation graph and moving it to the CPU. If `x`
    is a list or tuple, it will recursively convert each element to float.
    If `x` is of any other type, it will return `x` unchanged. Note that
    support for NumPy scalar arrays is yet to be implemented.

    Args:
        x (Any): The input value to be converted.

    Returns:
        Any: The converted float value, or the original input if conversion
        is not applicable.
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1: return float(x.detach().cpu())
    # TODO: numpy scalar arrays
    elif isinstance(x, (list, tuple)): return [ensure_float(i) for i in x]
    else: return x

class FreezeModel:
    def __init__(self, model:torch.nn.Module):
        self.original_requires_grads = []
        self.model = model
        for param in self.model.parameters():
            self.original_requires_grads.append(param.requires_grad)
            param.requires_grad = False

        self.frozen = True

    def unfreeze(self):
        """Unfreeze the parameters of the model.

        This method sets the `requires_grad` attribute of each parameter in the
        model to its original state, allowing gradients to be computed during
        backpropagation again. It is typically used to resume training after a
        period of freezing the parameters.
        """

        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = self.original_requires_grads[i]

        self.frozen = False


def is_container(mod:torch.nn.Module):
    """Determine if a PyTorch module is a container.

    This function checks whether the given module is a container by
    evaluating its children, parameters, and buffers. A module is considered
    a container if it has children but does not have any parameters or
    buffers. Containers are typically used to group other modules together
    without performing any computations themselves.

    Args:
        mod (torch.nn.Module): The PyTorch module to check.

    Returns:
        bool: True if the module is a container, False otherwise.
    """
    if len(list(mod.children())) == 0: return False # all containers have chilren
    if len(list(mod.parameters(False))) == 0 and len(list(mod.buffers(False))) == 0: return True # containers don't do anything themselves so they can't have parameters or buffers
    return False # has children, but has params or buffers

def param_count(module:torch.nn.Module): return sum(p.numel() for p in module.parameters())
def buffer_count(module:torch.nn.Module): return sum(b.numel() for b in module.buffers())


def _summary_hook(path:str, module:torch.nn.Module, input:tuple[torch.Tensor], output: torch.Tensor):
    """Generate and print a summary of the module's input and output.

    This function constructs a summary string that includes the path to the
    module, the type of the module, the sizes of the input tensors, the size
    of the output tensor, and some additional parameters related to the
    module. It formats this information into a structured output for easier
    debugging and analysis.

    Args:
        path (str): The path to the module.
        module (torch.nn.Module): The PyTorch module being summarized.
        input (tuple[torch.Tensor]): A tuple containing the input tensors.
        output (torch.Tensor): The output tensor from the module.

    Returns:
        None: This function does not return a value; it prints the summary directly.
    """
#pylint:disable=W0622
    input_info = '; '.join([(str(tuple(i.size())) if hasattr(i, "size") else str(i)[:100]) for i in input])
    print(
        f"{path:<45}{type_str(module):<45}{input_info:<25}{str(tuple(output.size())):<25}{param_count(module):<10}{buffer_count(module):<10}"
    )

def _register_summary_hooks(hooks:list, name:str, path:str, module:torch.nn.Module):
    """Register forward hooks for a PyTorch module to summarize its outputs.

    This function recursively traverses the children of the given PyTorch
    module and registers forward hooks that will call a summary function
    whenever the module processes input data. The hooks are stored in the
    provided list for later use. The path is constructed based on the
    module's hierarchy to uniquely identify each module in the summary.

    Args:
        hooks (list): A list to store the registered forward hooks.
        name (str): The name of the current module.
        path (str): The path to the current module in the hierarchy.
        module (torch.nn.Module): The PyTorch module for which hooks are being registered.

    Returns:
        None: This function does not return a value.
    """

    for name_, module_ in module.named_children():
        _register_summary_hooks(hooks, name_, f"{path}/{name}" if len(path)!=0 else name, module_)
    if not is_container(module):
        hooks.append(
            module.register_forward_hook(
                lambda m, i, o: _summary_hook(
                    f"{path}/{name}" if len(path) != 0 else name, m, i, o # type:ignore
                )
            )
        )

def summary(model: torch.nn.Module, input: Sequence | torch.Tensor, device:Any = CUDA_IF_AVAILABLE, orig_input = False, send_dummy=False):#pylint:disable=W0622
    """Print a summary table of the given PyTorch model.

    This function evaluates the model and prints a summary table that
    includes the path, module type, input size, output size, number of
    parameters, and number of buffers. It can also handle dummy inputs for
    the model to ensure that the summary is generated correctly without
    requiring actual data.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        input (Sequence | torch.Tensor): The input shape or tensor to the model.
        device (Any?): The device to which the model should be moved.
            Defaults to CUDA_IF_AVAILABLE.
        orig_input (bool?): If True, uses the original input tensor
            instead of generating a dummy input. Defaults to False.
        send_dummy (bool?): If True, sends a dummy input through the
            model to generate the summary. Defaults to False.

    Returns:
        None: This function does not return any value; it prints the summary
        directly.
    """
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        if send_dummy:
            if not orig_input:
                if isinstance(input, torch.Tensor): model(input.to(device))
                else: model(torch.randn(input, device = device))
            else: model(ensure_device(input, device))
        print(f"{'path':<45}{'module':<45}{'input size':<25}{'output size':<25}{'params':<10}{'buffers':<10}")

        hooks = []
        _register_summary_hooks(hooks, type_str(model), "", model)
        if not orig_input:
            if isinstance(input, torch.Tensor): model(input.to(device))
            else: model(torch.randn(input, device = device))
        else: model(ensure_device(input, device))
    for h in hooks: h.remove()


def one_batch(
    model: torch.nn.Module,
    inputs,
    targets,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device = CUDA_IF_AVAILABLE,
    train=True,
):
    """Perform a single training or evaluation batch.

    This function takes a model, inputs, and targets to compute the
    predictions and the loss. If training is enabled, it performs
    backpropagation and updates the model parameters using the provided
    optimizer. Optionally, it can also update the learning rate scheduler if
    one is provided. The function ensures that the inputs and targets are
    moved to the specified device (CPU or GPU) before processing.

    Args:
        model (torch.nn.Module): The neural network model to be trained or evaluated.
        inputs: The input data for the model.
        targets: The ground truth labels corresponding to the inputs.
        loss_fn (Callable): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]?): The learning rate scheduler to adjust the learning rate. Defaults to
            None.
        device: The device to run the model on (CPU or GPU). Defaults to CUDA if
            available.
        train (bool?): Flag indicating whether to perform training or evaluation.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - loss: The computed loss value.
            - preds: The predictions made by the model.
    """


    preds = model(ensure_device(inputs, device))
    loss = loss_fn(preds, ensure_device(targets, device))
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
    return loss, preds

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device = CUDA_IF_AVAILABLE,
        save_best = False
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.save_best = save_best
        if self.save_best:
            self.losses = []
            self.lowest_loss = float('inf')
            self.best_model = self.model.state_dict()

    def one_batch(self, inputs, targets, train = True):
        """Process a single batch of inputs and targets through the model.

        This function handles both training and evaluation modes for the model.
        When in training mode, it computes the loss, performs backpropagation,
        and updates the model parameters. In evaluation mode, it calculates the
        loss without updating the model. Additionally, it tracks the best model
        based on the lowest loss encountered during training.

        Args:
            inputs (torch.Tensor): The input data for the model.
            targets (torch.Tensor): The target labels corresponding to the inputs.
            train (bool?): A flag indicating whether to train the model.
                Defaults to True.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss for the current batch.
                - preds (torch.Tensor): The predictions made by the model for the
                inputs.
        """

        if train is False: self.model.eval()
        else: self.model.train()
        with nullcontext() if train else torch.no_grad():
            preds = self.model(ensure_device(inputs, self.device))
            loss = self.loss_fn(preds, ensure_device(targets, self.device))
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None: self.scheduler.step()
            if self.save_best:
                loss_value = loss.cpu().detach()
                if loss_value < self.lowest_loss:
                    self.lowest_loss = loss_value
                    self.best_model = self.model.state_dict()
                self.losses.append(loss_value)
            return loss, preds


def copy_state_dict(state_dict:dict):
    """Copy a state dictionary, cloning tensors and recursively copying nested
    dictionaries.

    This function takes a state dictionary as input and creates a new
    dictionary where each tensor is detached and cloned. If the value is a
    nested dictionary, it recursively calls itself to copy the inner
    dictionary. For other types of values, it uses the `try_copy` function
    to handle the copying process.

    Args:
        state_dict (dict): A dictionary containing state information, which may include tensors
            and nested dictionaries.

    Returns:
        dict: A new dictionary with cloned tensors and copied nested dictionaries.
    """

    return {
        k: (
            v.detach().clone()
            if isinstance(v, torch.Tensor)
            else copy_state_dict(v)
            if isinstance(v, dict)
            else try_copy(v)
        )
        for k, v in try_copy(state_dict.items())
    }


class BackupModule:
    def __init__(self, model:torch.nn.Module | Any):
        self.model = model
        self.state_dict = copy_state_dict(model.state_dict())

    def update(self, model:Optional[torch.nn.Module] = None):
        """Update the state dictionary of the current model.

        This method updates the internal state dictionary with the state
        dictionary of the provided model. If no model is provided, it defaults
        to using the instance's current model. This is useful for synchronizing
        the state of different models or for saving/loading model states.

        Args:
            model (Optional[torch.nn.Module]): The model whose state dictionary
        """

        if model is None: model = self.model
        self.state_dict = copy_state_dict(model.state_dict()) # type:ignore

    def restore(self, model:Optional[torch.nn.Module] = None):
        """Restore a model's state from a saved state dictionary.

        This function loads the state dictionary into the specified model. If no
        model is provided, it uses the instance's model. The state dictionary is
        copied from the current object's state dictionary to ensure that the
        model is restored to its previous state.

        Args:
            model (Optional[torch.nn.Module]): The model to restore. If None,

        Returns:
            torch.nn.Module: The model with the restored state.
        """

        if model is None: model = self.model
        model.load_state_dict(copy_state_dict(self.state_dict)) # type:ignore
        return model


def get_lr(optimizer:torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer:torch.optim.Optimizer, lr:int|float):
    """Set the learning rate for an optimizer.

    This function updates the learning rate of all parameter groups in the
    given optimizer to the specified value. It iterates over each parameter
    group in the optimizer and sets the learning rate accordingly.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate
            needs to be updated.
        lr (int | float): The new learning rate to be set for the optimizer.
    """

    for g in optimizer.param_groups:
        g['lr'] = lr

def change_lr(optimizer:torch.optim.Optimizer, fn:Callable):
    """Change the learning rate of an optimizer.

    This function iterates over the parameter groups of the given optimizer
    and applies a provided function to modify the learning rate of each
    parameter group. This can be useful for implementing learning rate
    schedules or adjustments during training.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rates
            are to be changed.
        fn (Callable): A function that takes the current learning rate as input
            and returns the modified learning rate.

    Returns:
        None: This function modifies the optimizer in place and does not return
            any value.
    """

    for g in optimizer.param_groups:
        g['lr'] = fn(g['lr'])

def lr_finder_fn(
    one_batch_fn: Callable,
    optimizer: torch.optim.Optimizer,
    dl: torch.utils.data.DataLoader | Iterable,
    start=1e-6,
    mul=1.3,
    add=0,
    end=1,
    max_increase:Optional[float|int]=3,
    plot=True,
    log = True,
    device: Any = CUDA_IF_AVAILABLE,
):
    """Find the optimal learning rate for training a model.

    This function implements a learning rate finder that helps in
    identifying the optimal learning rate for training a model. It does this
    by gradually increasing the learning rate from a specified starting
    point and monitoring the loss at each step. The learning rate is
    adjusted according to the specified multiplication factor and addition
    value. The process continues until either the learning rate exceeds a
    specified maximum value or the loss increases by a certain factor
    compared to the minimum loss observed.

    Args:
        one_batch_fn (Callable): A function that takes inputs and targets,
            and returns the loss and other metrics for a single batch.
        optimizer (torch.optim.Optimizer): The optimizer used for updating
            the model parameters.
        dl (torch.utils.data.DataLoader | Iterable): The data loader or
            iterable providing batches of data for training.
        start (float?): The initial learning rate. Defaults to 1e-6.
        mul (float?): The factor by which to multiply the learning
            rate at each iteration. Defaults to 1.3.
        add (float?): The value to add to the learning rate at each
            iteration. Defaults to 0.
        end (float?): The maximum learning rate to reach. Defaults to 1.
        max_increase (Optional[float|int]?): The maximum allowed
            increase in loss compared to the minimum loss observed. Defaults to 3.
        plot (bool?): Whether to plot the learning rates against
            losses. Defaults to True.
        log (bool?): Whether to log the learning rates and losses
            during the process. Defaults to True.
        device (Any?): The device on which to perform computations.
            Defaults to CUDA if available.

    Returns:
        tuple: A tuple containing two lists - the learning rates and the
        corresponding losses.

    Raises:
        ValueError: If neither `end` nor `max_increase` is specified.
    """

    if device is None: device = torch.device('cpu')
    lrs = []
    losses = []
    set_lr(optimizer, start)
    if end is None and max_increase is None: raise ValueError("Specify at least one of `end` or `max_increase`.")
    converged = False
    dl_iter = EndlessContinuingIterator(dl)
    while True:
        for inputs, targets in dl_iter:

            loss, _ = one_batch_fn(ensure_device(inputs,device), ensure_device(targets,device), train=True)
            loss = float(loss.detach().cpu())
            lrs.append(get_lr(optimizer))
            losses.append(loss)

            change_lr(optimizer, lambda x: x * mul + add)

            if log:print(f"lr: {get_lr(optimizer)} loss: {loss}", end="\r")
            if (end is not None and get_lr(optimizer) > end) or (max_increase is not None and loss/min(losses) > max_increase):
                converged = True
                break

        if converged: break

    if plot:
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.show()
    return lrs, losses

def lr_finder(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    dl: torch.utils.data.DataLoader | Iterable,
    start=1e-6,
    mul=1.3,
    add=0,
    end=1,
    max_increase:Optional[float|int]=3,
    niter=1,
    return_best = False,
    plot=True,
    log = True,
    device: Any = CUDA_IF_AVAILABLE,
) -> tuple:
    """Find the optimal learning rate for a given model and dataset.

    This function implements a learning rate finder that helps in
    determining the best learning rate for training a model. It gradually
    increases the learning rate from a specified starting point to a maximum
    value, while tracking the loss at each step. The function can run for
    multiple iterations and can return the model with the lowest loss if
    specified. Additionally, it can plot the learning rates against the
    average losses to visualize the optimal learning rate.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss_fn (Callable): The loss function used to compute the loss.
        dl (torch.utils.data.DataLoader | Iterable): The data loader or iterable providing the training data.
        start (float?): The starting learning rate. Defaults to 1e-6.
        mul (float?): The factor by which to multiply the learning rate at each step. Defaults
            to 1.3.
        add (float?): A constant value to add to the learning rate at each step. Defaults to
            0.
        end (float?): The maximum learning rate to reach. Defaults to 1.
        max_increase (Optional[float | int]?): The maximum number of times the loss can increase before stopping.
            Defaults to 3.
        niter (int?): The number of iterations to run the learning rate finder. Defaults to 1.
        return_best (bool?): Whether to return the model with the lowest loss. Defaults to False.
        plot (bool?): Whether to plot the learning rates against losses. Defaults to True.
        log (bool?): Whether to log progress during execution. Defaults to True.
        device (Any?): The device on which to perform computations. Defaults to
            CUDA_IF_AVAILABLE.

    Returns:
        tuple: A tuple containing either the model and lists of learning rates and
            average losses if `return_best` is True,
            or just lists of learning rates and average losses if False.
    """

    iter_losses:list[list[float]] = []
    iter_lrs:list[list[float]] = []

    if return_best:
        lowest_loss = float("inf")
        best_model = None
    try:
        for _ in range(niter):
            model_backup = BackupModule(model) if hasattr(model, "state_dict") else None
            optimizer_backup = BackupModule(optimizer) if hasattr(optimizer, "state_dict") else None
            model.train()
            trainer = Trainer(model, loss_fn, optimizer, device = device, save_best=return_best)
            fn = trainer.one_batch
            lrs, losses = lr_finder_fn(
                one_batch_fn=fn,
                optimizer=optimizer,
                dl=dl,
                start=start,
                mul=mul,
                add=add,
                end=end,
                max_increase=max_increase,
                plot=False,
                device=device,
            )
            iter_losses.append(losses[:-1])
            iter_lrs.append(lrs[:-1])

            if return_best:
                if trainer.lowest_loss < lowest_loss: # type:ignore
                    lowest_loss = trainer.lowest_loss
                    best_model = trainer.best_model

            if model_backup is not None: model_backup.restore()
            if optimizer_backup is not None: optimizer_backup.restore()
            if log:print(f"Iteration {_} done.", end = '\r')

    except KeyboardInterrupt: pass
    avg_losses = [[j for j in i if j is not None] for i in zip_longest(*iter_losses)]
    avg_losses = [sum(i)/len(i) for i in avg_losses]
    lrs = [i[0] for i in zip_longest(*iter_lrs)]
    if log:print()
    if plot:
        plt.plot(lrs, avg_losses)
        plt.xscale('log')
        plt.show()
    if return_best:
        model.load_state_dict(best_model) # type:ignore
        return model, lrs, avg_losses
    else:
        return lrs, avg_losses


def has_nonzero_weight(mod:torch.nn.Module): return hasattr(mod, "weight") and mod.weight.std!=0

def apply_init_fn(model:torch.nn.Module, init_fn: Callable, filt = has_nonzero_weight) -> torch.nn.Module:
    return model.apply(lambda m: init_fn(m.weight) if hasattr(m, "weight") and (filt(m) if filt is not None else True) else None)

def smart_tonumpy(t):
    """Convert a PyTorch tensor to a NumPy array.

    This function checks if the input is a PyTorch tensor. If it is, the
    tensor is detached from the current computation graph, moved to the CPU,
    and then converted to a NumPy array. If the input is not a tensor, it is
    returned unchanged.

    Args:
        t (torch.Tensor or any): The input that may be a PyTorch tensor.

    Returns:
        numpy.ndarray or any: The converted NumPy array if the input
        was a tensor, otherwise the original input.
    """

    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return t


def to_binary(t:torch.Tensor, threshold:float = 0.5):
    return torch.where(t > threshold, 1, 0)


def center_of_mass(feature:torch.Tensor):
    """Compute the center of mass of a 4D or 5D tensor.

    This function calculates the center of mass (COM) of the input tensor,
    which can represent a batch of images in either 4D or 5D format. The COM
    is computed by summing the weighted coordinates of the pixels in the
    tensor and normalizing by the total mass. The function supports tensors
    with the following shapes: - 5D tensor: [batch, x, y, z, channel] - 4D
    tensor: [batch, x, y, channel]  The function uses PyTorch operations to
    perform the calculations efficiently.

    Args:
        feature (torch.Tensor): Input tensor representing images. It should be a 5D tensor
            with shape [batch, x, y, z, channel] or a 4D tensor with
            shape [batch, x, y, channel].

    Returns:
        torch.Tensor: A tensor containing the center of mass coordinates for each image in
            the input batch.

    Raises:
        NotImplementedError: If the input tensor has an unsupported number of dimensions.
    """
    if feature.ndim == 3: nx, ny, nz = feature.shape
    elif feature.ndim == 2: nx, ny = feature.shape
    else: raise NotImplementedError
    map1 = feature.unsqueeze(0).unsqueeze(-1)
    n_dim = map1.ndim

    if n_dim == 5:
        x = torch.sum(map1, dim =(2,3))
    else:
        x = torch.sum(map1, dim = 2)

    r1 = torch.arange(0,nx, dtype = torch.float32)
    r1 = torch.reshape(r1, (1,nx,1))

    x_product = x*r1
    x_weight_sum = torch.sum(x_product,dim = 1,keepdim=True)+0.00001
    x_sum = torch.sum(x,dim = 1,keepdim=True)+0.00001
    cm_x = torch.divide(x_weight_sum,x_sum)

    if n_dim == 5:
        y = torch.sum(map1, dim =(1,3))
    else:
        y = torch.sum(map1, dim = 1)

    r2 = torch.arange(0,ny, dtype = torch.float32)
    r2 = torch.reshape(r2, (1,ny,1))

    y_product = y*r2
    y_weight_sum = torch.sum(y_product,dim = 1,keepdim=True)+0.00001
    y_sum = torch.sum(y,dim = 1,keepdim=True)+0.00001
    cm_y = torch.divide(y_weight_sum,y_sum)

    if n_dim == 5:
        z = torch.sum(map1, dim =(1,2))

        r3 = torch.arange(0,nz, dtype = torch.float32) # type:ignore
        r3 = torch.reshape(r3, (1,nz,1)) # type:ignore

        z_product = z*r3
        z_weight_sum = torch.sum(z_product,dim = 1,keepdim=True)+0.00001
        z_sum = torch.sum(z,dim = 1,keepdim=True)+0.00001
        cm_z = torch.divide(z_weight_sum,z_sum)

        center_mass = torch.concat([cm_x,cm_y,cm_z],dim=1)
    else:
        center_mass = torch.concat([cm_x,cm_y],dim=1)

    return center_mass[0].squeeze(1)

def binary_erode3d(tensor, n = 1):
    """Erodes a 3D binary tensor.

    This function performs a morphological erosion operation on a 3D binary
    tensor. The erosion is applied using a predefined kernel, and the
    operation can be repeated multiple times based on the parameter `n`. If
    `n` is greater than 1, the function recursively calls itself to apply
    the erosion multiple times.

    Args:
        tensor (torch.Tensor): A 3D binary tensor to be eroded.
        n (int?): The number of times to apply the erosion. Defaults to 1.

    Returns:
        torch.Tensor: A 3D binary tensor after applying the erosion operation.
    """
    if n > 1: tensor = binary_erode3d(tensor, n-1)
    kernel = torch.tensor([[[[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]]], dtype=torch.int64)
    convolved = torch.nn.functional.conv3d(input = tensor.unsqueeze(0), weight = kernel, padding=1) # pylint:disable=E1102
    return torch.where(convolved==7, 1, 0)[0]



def area_around(tensor:torch.Tensor, coord, size) -> torch.Tensor:
    """Returns a tensor of specified size around given coordinates.

    This function extracts a sub-tensor from the input tensor based on the
    provided coordinates and size. The coordinates can be either 2D or 3D,
    and the function adjusts the coordinates to ensure that the extracted
    area remains within the bounds of the original tensor. The size
    parameter determines the extent of the area to be extracted around the
    specified coordinates.

    Args:
        tensor (torch.Tensor): The input tensor from which to extract the area.
        coord (tuple): A tuple representing the coordinates around which to extract
            the area. It can be either 2D (x, y) or 3D (x, y, z).
        size (tuple): A tuple representing the size of the area to extract. For
            2D coordinates, it should be (sx, sy), and for 3D coordinates, it
            should be (sx, sy, sz).

    Returns:
        torch.Tensor: A tensor containing the extracted area around the specified
        coordinates.

    Raises:
        NotImplementedError: If the number of dimensions of the tensor is not
    """
    if len(coord) == 3:
        x, y, z = coord
        x, y, z = int(x), int(y), int(z)
        sx, sy, sz = size
        sx, sy, sz = int(sx//2), int(sy//2), int(sz//2)
        if tensor.ndim == 3: shape = tensor.size()
        elif tensor.ndim == 4: shape = tensor.shape[1:]
        else: raise NotImplementedError

        if x-sx < 0: x = x - (x-sx)
        if y-sy < 0: y = y - (y-sy)
        if z-sz < 0: z = z - (z-sz)
        if x+sx+1 > shape[0]: x = x - (x+sx+1 - shape[0])
        if y+sy+1 > shape[1]: y = y - (y+sy+1 - shape[1])
        if z+sz+1 > shape[2]: z = z - (z+sz+1 - shape[2])
        if tensor.ndim == 3: return tensor[int(x-sx):int(x+sx), int(y-sy):int(y+sy), int(z-sz):int(z+sz)]
        elif tensor.ndim == 4:
            return tensor[:, int(x-sx):int(x+sx), int(y-sy):int(y+sy), int(z-sz):int(z+sz)]
        else: raise NotImplementedError

    elif len(coord) == 2:
        x, y = coord
        sx, sy = size
        sx, sy = int(sx/2), int(sy/2)
        if tensor.ndim == 2: shape = tensor.size()
        elif tensor.ndim == 3: shape = tensor.shape[1:]
        elif tensor.ndim == 4: shape = tensor.shape[2:]
        else: raise NotImplementedError

        if x-sx < 0: x = x - (x-sx)
        if y-sy < 0: y = y - (y-sy)
        if x+sx+1 > shape[0]: x = x - (x+sx+1 - shape[0])
        if y+sy+1 > shape[1]: y = y - (y+sy+1 - shape[1])
        if tensor.ndim == 2: return tensor[int(x-sx):int(x+sx), int(y-sy):int(y+sy)]
        elif tensor.ndim == 3:
            return tensor[:, int(x-sx):int(x+sx), int(y-sy):int(y+sy)]
        elif tensor.ndim == 4:
            return tensor[:,:, int(x-sx):int(x+sx), int(y-sy):int(y+sy)]
        else: raise NotImplementedError
    else: raise NotImplementedError


def one_hot_mask(mask: torch.Tensor, num_classes:int) -> torch.Tensor:
    """Generate a one-hot encoded mask from the input tensor.

    This function takes a tensor representing class indices and converts it
    into a one-hot encoded format. The output tensor will have an additional
    dimension for the classes, where each class index in the input tensor is
    represented as a one-hot vector. The function handles both 2D and 3D
    input tensors. For 3D tensors, the output will have the shape
    (num_classes, depth, height, width), while for 2D tensors, the output
    will have the shape (num_classes, height, width).

    Args:
        mask (torch.Tensor): A tensor containing class indices. It can be either 2D or 3D.
        num_classes (int): The total number of classes for one-hot encoding.

    Returns:
        torch.Tensor: A one-hot encoded tensor with an additional dimension for classes.

    Raises:
        NotImplementedError: If the input tensor has an unsupported number of dimensions.
    """

    if mask.ndim == 3:
        return torch.nn.functional.one_hot(mask.to(torch.int64), num_classes).permute(3, 0, 1, 2).to(torch.float32) # pylint:disable=E1102 #type:ignore
    elif mask.ndim == 2:
        return torch.nn.functional.one_hot(mask.to(torch.int64), num_classes).permute(2, 0, 1).to(torch.float32) # pylint:disable=E1102 #type:ignore
    else: raise NotImplementedError(f'one_hot_mask: mask.ndim = {mask.ndim}')


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def replace_layers(model:torch.nn.Module, old:type, new:torch.nn.Module):
    """Replace specified layers in a PyTorch model.

    This function recursively traverses a given PyTorch model and replaces
    all instances of a specified layer type with a new layer type. It checks
    each child module of the model, and if the module is of the specified
    old type, it replaces it with the new module using `setattr`.

    Args:
        model (torch.nn.Module): The PyTorch model containing layers to be replaced.
        old (type): The type of the layer to be replaced.
        new (torch.nn.Module): The new layer that will replace the old layer.

    Returns:
        None: This function modifies the model in place and does not return a value.
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)

def replace_conv(model:torch.nn.Module, old:type, new:type):
    """Replace instances of a specific convolutional layer in a model.

    This function traverses the given PyTorch model and replaces all
    instances of a specified convolutional layer type (`old`) with a new
    layer type (`new`). It recursively checks each child module of the
    model, ensuring that all nested layers are also examined and replaced if
    they match the specified type. The new layer is initialized with the
    same parameters as the old layer.

    Args:
        model (torch.nn.Module): The PyTorch model containing convolutional layers.
        old (type): The type of the convolutional layer to be replaced.
        new (type): The new convolutional layer type to replace the old one.

    Returns:
        None: This function modifies the model in place and does not return a value.
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_conv(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new(module.in_channels, module.out_channels, module.kernel_size,
                                  module.stride, module.padding, module.dilation, module.groups))

def replace_conv_transpose(model:torch.nn.Module, old:type, new:type):
    """Replace instances of a specific convolutional transpose layer in a
    model.

    This function traverses the given PyTorch model and replaces all
    instances of the specified old convolutional transpose layer type with a
    new layer type. It maintains the original parameters of the layers being
    replaced, ensuring that the new layers are initialized with the same
    configuration as the old ones. The function also handles nested modules
    by recursively calling itself on compound modules.

    Args:
        model (torch.nn.Module): The PyTorch model containing layers to be replaced.
        old (type): The type of the convolutional transpose layer to be replaced.
        new (type): The type of the new convolutional transpose layer to replace the old
            one.
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_conv(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new(module.in_channels, module.out_channels, module.kernel_size,
                                  module.stride, module.padding, module.output_padding, module.groups, True, module.dilation))


def unonehot(mask: torch.Tensor, batch = False) -> torch.Tensor:
    """Convert a one-hot encoded tensor back to class indices.

    This function takes a one-hot encoded tensor and returns the indices of
    the maximum values along the specified dimension. If the `batch`
    parameter is set to True, it will return the indices for each batch
    along dimension 1; otherwise, it will return the indices along dimension
    0.

    Args:
        mask (torch.Tensor): A one-hot encoded tensor.
        batch (bool?): A flag indicating whether to process in batch mode. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the indices of the maximum values.
    """

    if batch: return torch.argmax(mask, dim=1)
    return torch.argmax(mask, dim=0)


def preds_batch_to_onehot(preds:torch.Tensor):
    return one_hot_mask(preds.argmax(1), preds.shape[1]).swapaxes(0,1)


def angle(a, b, dim=-1):
    """Calculate the angle between two tensors.

    This function computes the angle between two tensors `a` and `b` using
    the formula derived from the norms of the tensors. It normalizes the
    tensors and applies the arctangent function to determine the angle. The
    computation is performed along a specified dimension, which defaults to
    -1 (the last dimension).

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.
        dim (int?): The dimension along which to compute the angle.
            Defaults to -1.

    Returns:
        torch.Tensor: The computed angle(s) between the input tensors in radians.
    """
    a_norm = a.norm(dim=dim, keepdim=True)
    b_norm = b.norm(dim=dim, keepdim=True)
    return 2 * torch.atan2(
        (a * b_norm - a_norm * b).norm(dim=dim),
        (a * b_norm + a_norm * b).norm(dim=dim)
    )

@contextmanager
def seeded_rng(seed:Optional[Any]=0):
    """Context manager for setting seeds for random number generators.

    This context manager sets the seed for the random number generators in
    PyTorch, NumPy, and the built-in Python random module. If the provided
    seed is None, the context manager does nothing and yields control
    without modifying any states. When the context manager is exited, it
    restores the original random states for all three libraries.

    Args:
        seed (Optional[Any]): The seed value to set for the random number

    Yields:
        None: Control is yielded back to the context in which this
        manager is used.
    """
    if seed is None:
        yield
        return
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    yield
    torch.random.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)

def seed0_worker(worker_id):
    """Set the random seed for a worker process.

    This function initializes the random seed for the worker process using
    the initial seed from PyTorch. It ensures that the random number
    generation is consistent across different runs by seeding NumPy and
    Python's built-in random module with the same value derived from the
    worker's initial seed.

    Args:
        worker_id (int): The identifier for the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed0_generator = torch.Generator()
seed0_generator.manual_seed(0)

seed0_kwargs = {'generator': seed0_generator, 'worker_init_fn': seed0_worker}
"""Kwargs for pytorch dataloader so that it is deterministic"""

def seeded_randperm(n,
    *,
    out = None,
    dtype= None,
    layout = None,
    device= None,
    pin_memory = False,
    requires_grad = False,
    seed=0,
    ):
    """Generate a random permutation of integers from 0 to n-1 using a
    specified seed.

    This function utilizes a seeded random number generator to produce a
    random permutation of integers. The parameters allow for customization
    of the output tensor, including its data type, layout, device, and
    whether it requires gradients. The seed ensures that the random
    permutation can be reproduced.

    Args:
        n (int): The upper limit of the random permutation (exclusive).
        out (Tensor?): The output tensor to store the result. Defaults to None.
        dtype (torch.dtype?): The desired data type of the output tensor. Defaults to None.
        layout (torch.layout?): The desired layout of the output tensor. Defaults to None.
        device (torch.device?): The device on which to allocate the output tensor. Defaults to None.
        pin_memory (bool?): If True, the returned tensor will be pinned in memory. Defaults to
            False.
        requires_grad (bool?): If True, gradients will be tracked for the output tensor. Defaults to
            False.
        seed (int?): The seed for the random number generator. Defaults to 0.

    Returns:
        Tensor: A tensor containing a random permutation of integers from 0 to n-1.
    """

    with seeded_rng(seed):
        return torch.randperm(n, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)

def stepchunk(vec:torch.Tensor|np.ndarray, chunks:int, maxlength:Optional[int]=None):
    """Split a tensor or ndarray into chunks.

    This function takes a tensor or a NumPy array and divides it into
    smaller chunks based on the specified number of chunks. The maximum
    length of each chunk can be specified; if not provided, it defaults to
    the length of the input vector. The resulting chunks are created by
    slicing the input vector at regular intervals determined by the number
    of chunks.

    Args:
        vec (torch.Tensor | np.ndarray): The input tensor or NumPy array to be chunked.
        chunks (int): The number of chunks to divide the input vector into.
        maxlength (Optional[int]): The maximum length of each chunk. If None, defaults to the length of
            `vec`.

    Returns:
        list: A list containing the resulting chunks of the input vector.
    """

    maxlength = maxlength or vec.shape[0]
    return [vec[i : i+maxlength : chunks] for i in range(chunks)]

class ConcatZeroChannelsToDataloader:
    """Wraps dataloader and adds zero channels to the end, useful when model accepts more channels than images have"""
    def __init__(self, dataloader, resulting_channels):
        self.dataloader = dataloader
        self.resulting_channels=resulting_channels
    def __len__(self): return len(self.dataloader)
    def __iter__(self):
        """Iterate over the dataloader to yield modified inputs and targets.

        This method modifies the input tensors by concatenating zeros to the
        second dimension based on the difference between the resulting channels
        and the current number of channels in the input tensors. It yields the
        modified inputs along with their corresponding targets for further
        processing.

        Yields:
            tuple: A tuple containing the modified inputs and their corresponding targets.
        """

        for inputs, targets in self.dataloader:
            shape = list(inputs.shape)
            shape[1] = self.resulting_channels - shape[1]
            inputs = torch.cat((inputs, torch.zeros(shape)), dim=1)
            yield inputs, targets

class BatchInputTransforms:
    """Wraps dataloader and applies transforms to batch inputs. So don't use stuff like randflip."""
    def __init__(self, dataloader, transforms):
        self.dataloader = dataloader
        self.transforms = Compose(transforms)
    def __len__(self): return len(self.dataloader)
    def __iter__(self):
        """Iterate over the dataloader to yield transformed inputs and targets.

        This method allows iteration over the dataset by yielding pairs of
        transformed inputs and their corresponding targets. It utilizes the
        dataloader to fetch the data and applies the specified transformations
        to the inputs before yielding them.

        Yields:
            tuple: A tuple containing the transformed inputs and their
            corresponding targets.
        """

        for inputs, targets in self.dataloader:
            yield self.transforms(inputs), targets

def map_to_base_np(number:int, base):
    """Convert an integer into a list of digits of that integer in a given
    base.

    This function takes an integer and converts it into its representation
    in a specified base. It handles the conversion by calculating the digits
    of the integer in the provided base and returns them as a NumPy array.
    If the input number is zero, it returns zero directly.

    Args:
        number (int): The integer to convert.
        base (int): The base to convert the integer to.

    Returns:
        numpy.ndarray: An array of digits representing the input integer in the given base.
    """
    if number == 0: return 0
    # Convert the input numbers to their digit representation in the given base
    digits = np.array([number])
    base_digits = (digits // base**(np.arange(int(np.log(number) / np.log(base)) + 1)[::-1])) % base

    return base_digits

def map_to_base(number:int, base):
    """Convert an integer into a list of digits of that integer in a given
    base.

    This function takes an integer and converts it into its representation
    in a specified base. It handles the conversion by calculating the digits
    of the integer in the given base and returns them as a tensor. If the
    input number is zero, it directly returns a tensor containing zero. The
    conversion is performed using integer division and modulus operations.

    Args:
        number (int): The integer to convert.
        base (int): The base to convert the integer to.

    Returns:
        torch.Tensor: A tensor of digits representing the input integer in the given base.
    """
    if number == 0: return torch.tensor([0])
    # Convert the input numbers to their digit representation in the given base
    digits = torch.tensor([number])
    base_digits = (digits // base**(torch.arange(int(math.log(number) / math.log(base)), -1, -1))) % base

    return base_digits


def sliding_inference_around_3d(input:torch.Tensor, inferer, size, step, around, nlabels):
    """Perform sliding inference on a 3D tensor.

    This function applies a sliding window approach to perform inference on
    a 3D tensor. It processes the input tensor by extracting patches around
    a specified center point and applying the provided inference function.
    The results are accumulated and averaged over the number of patches
    processed.

    Args:
        input (torch.Tensor): A 4D or 5D tensor representing the input data.
        inferer: A callable that takes a tensor as input and returns predictions.
        size (tuple): A tuple specifying the size of the patch to extract.
        step (int): The step size for moving the sliding window.
        around (int): The number of slices to consider around the center slice.
        nlabels (int): The number of labels or channels in the output.

    Returns:
        torch.Tensor: A tensor containing the averaged predictions for each
        position in the input tensor.
    """
    if input.ndim == 4: input = input.unsqueeze(0)
    results = torch.zeros((input.shape[0], nlabels, *input.shape[2:]), device=input.device,)
    counts = torch.zeros_like(results)
    for x in range(around, input.shape[2]-around, 1):
        for y in range(0, input.shape[3], step):
            for z in range(0, input.shape[4], step):
                preds = inferer(input[:, :, x-1:x+around+1, y:y+size[0], z:z+size[1]])
                results[:, :, x, y:y+size[0], z:z+size[1]] += preds
                counts[:, :, x, y:y+size[0], z:z+size[1]] += 1

    results /= counts
    return results


class CreateIterator:
    def __init__(self, iterable:Iterable, length: int):
        self.iterable = iterable
        self.length = length
    def __len__(self): return self.length
    def __iter__(self): return self.iterable

class MRISlicer:
    def __init__(self, tensor:torch.Tensor, seg:torch.Tensor, num_classes:int, around:int = 1, any_prob:float = 0.05, warn_empty = True):
        if tensor.ndim != 4: raise ValueError(f"`tensor` is {tensor.shape}")
        if seg.ndim not in (3, 4): raise ValueError(f"`seg` is {seg.shape}")
        if seg.ndim == 4: seg = seg.argmax(0)

        self.tensor = tensor
        self.seg = seg
        self.num_classes = num_classes

        if self.tensor.shape[1:] != self.seg.shape: raise ValueError(f"Shapes don't match: image is {self.tensor.shape}, seg is {self.seg.shape}")

        self.x,self.y,self.z = [],[],[]

        # save top
        for i, sl in enumerate(to_binary(seg, 0)):
            if sl.sum() > 0: self.x.append(i)

        # save front
        for i, sl in enumerate(to_binary(seg.swapaxes(0,1), 0)):
            if sl.sum() > 0: self.y.append(i)

        # save side
        for i, sl in enumerate(to_binary(seg.swapaxes(0,2), 0)):
            if sl.sum() > 0: self.z.append(i)

        if len(self.x) == 0:
            if warn_empty: logging.warning('Segmentation is empty, setting probability to 0.')
            self.any_prob = 0

        self.shape = self.tensor.shape
        self.around = around
        self.any_prob = any_prob

    def set_settings(self, around:Optional[int] = None, any_prob: Optional[float] = None):
        """Set the settings for the instance.

        This method updates the instance variables `around` and `any_prob` based
        on the provided arguments. If `around` is not None, it updates the
        `around` attribute. Additionally, if the list `x` is not empty and
        `any_prob` is provided, it updates the `any_prob` attribute.

        Args:
            around (Optional[int]): An optional integer value to set the `around` attribute.
            any_prob (Optional[float]): An optional float value to set the `any_prob` attribute,
                only if the list `x` is not empty.
        """

        if around is not None: self.around = around
        if len(self.x) > 0 and any_prob is not None: self.any_prob = any_prob

    def __call__(self):
        """Select a random coordinate from a specified dimension.

        This method randomly chooses a dimension (0, 1, or 2) and then selects a
        coordinate based on that dimension. If a random condition is met, it
        picks a coordinate from predefined lists; otherwise, it generates a
        random coordinate within a specified range. The selected coordinate is
        then used to retrieve a slice of data.

        Returns:
            The result of the `get_slice` method, which is called with the selected
                dimension
            and coordinate.
        """

        # pick a dimension
        dim: Literal[0,1,2] = random.choice([0,1,2])

        # get length
        if dim == 0: length = self.shape[1]
        elif dim == 1: length = self.shape[2]
        else: length = self.shape[3]

        # pick a coord
        # from segmentation
        if random.random() > self.any_prob:
            if dim == 0: coord = random.choice(self.x)
            elif dim == 1: coord = random.choice(self.y)
            else: coord = random.choice(self.z)

        else:
            coord = random.randrange(self.around, length - self.around)

        return self.get_slice(dim, coord)

    def get_slice(self, dim: Literal[0,1,2], coord: int):
        """Get a slice from the tensor based on the specified dimension and
        coordinate.

        This method retrieves a slice of the tensor and its corresponding
        segmentation based on the provided dimension (`dim`) and coordinate
        (`coord`). The function handles different dimensions by swapping axes as
        necessary. It also ensures that the coordinate is within valid bounds by
        adjusting it if it falls outside the specified range. Depending on the
        value of `around`, it either returns a single slice or a slice of values
        around the specified coordinate, potentially flipping the slice based on
        a random condition.

        Args:
            dim (Literal[0, 1, 2]): The dimension from which to get the slice.
            coord (int): The coordinate index for slicing.

        Returns:
            tuple: A tuple containing the sliced tensor and the corresponding segmentation.
        """
        # get a tensor
        if dim == 0:
            tensor = self.tensor
            seg = self.seg
            length = self.shape[1]
        elif dim == 1:
            tensor = self.tensor.swapaxes(1, 2)
            seg = self.seg.swapaxes(0,1)
            length = self.shape[2]
        else:
            tensor = self.tensor.swapaxes(1, 3)
            seg = self.seg.swapaxes(0,2)
            length = self.shape[3]

        # check if coord outside of bounds
        if coord < self.around: coord = self.around
        elif coord + self.around >= length: coord = length - self.around - 1


        # get slice
        if self.around == 0: return tensor[:, coord], seg[coord]

        # or get slices around (and flip slice spatial dimension with 0.5 p)
        if random.random() > 0.5: return tensor[:, coord - self.around : coord + self.around + 1].flatten(0,1), seg[coord]
        return tensor[:, coord - self.around : coord + self.around + 1].flip((1,)).flatten(0,1), seg[coord]

    def get_random_slice(self):
        """Get a random slice, ignoring the `any_prob` parameter.

        This method selects a random dimension from the available dimensions of
        the object's shape and retrieves a slice from that dimension. The length
        of the slice is determined based on the selected dimension, and a random
        coordinate is generated within the specified bounds, adjusted by the
        `around` attribute. The resulting slice is obtained by calling the
        `get_slice` method with the chosen dimension and coordinate.

        Returns:
            Slice: A random slice from the selected dimension.
        """
        # pick a dimension
        dim: Literal[0,1,2] = random.choice([0,1,2])

        # get length
        if dim == 0: length = self.shape[1]
        elif dim == 1: length = self.shape[2]
        else: length = self.shape[3]

        coord = random.randrange(0 + self.around, length - self.around)
        return self.get_slice(dim, coord)

    def yield_all_seg_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slice callables for segmentation.

        This function iterates over three dimensions (0, 1, and 2) and yields
        callable functions that, when invoked, will return a tuple of
        torch.Tensor objects representing the slices for the specified dimension
        and coordinate. The coordinates are obtained from the instance variables
        `self.x`, `self.y`, and `self.z` corresponding to each dimension.

        Yields:
            Callable[[], tuple[torch.Tensor, torch.Tensor]]: A callable that
            returns a tuple of tensors representing the slice for a given
            dimension and coordinate.
        """
        # pick a dimension
        for dim in (0, 1, 2):

            if dim == 0: coord_list = self.x
            elif dim == 1: coord_list = self.y
            else: coord_list = self.z

            for coord in coord_list:

                yield functools.partial(self.get_slice, dim, coord)

    def get_all_seg_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Retrieve all callable slices that yield segmentation as partials.

        This function collects and returns a list of callables that, when
        invoked, will produce tuples containing two torch.Tensor objects. These
        callables are specifically designed to handle segmentation tasks within
        the context of the application.

        Returns:
            list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]: A list of callables
            that yield tuples of torch.Tensor objects.
        """
        return list(self.yield_all_seg_slice_callables())

    def get_all_seg_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve all segmentation slices.

        This function collects and returns all slices that have associated
        segmentation information. It does so by invoking all callable objects
        that are responsible for generating these segmentation slices. The
        result is a list of tuples, where each tuple contains the segmentation
        data in the form of PyTorch tensors.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor]]: A list of tuples, each
            containing segmentation slices as PyTorch tensors.
        """
        return [i() for i in self.get_all_seg_slice_callables()]

    def yield_all_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slice callables for a given tensor dimension.

        This function generates callable functions that return slices of a
        tensor for each dimension (0, 1, and 2). It includes empty segmentation
        slices as well. The length of the slices is determined based on the
        specified dimension, and the function yields partial functions that can
        be called to obtain the corresponding tensor slices.

        Yields:
            Callable[[], tuple[torch.Tensor, torch.Tensor]]: A callable that, when invoked,
            returns a tuple of tensors corresponding to the specified slice.

        Note:
            The function assumes that `self.shape` and `self.around` are defined
            attributes of the class instance.
        """
        # pick a dimension
        for dim in (0, 1, 2):

            # get length
            if dim == 0: length = self.shape[1]
            elif dim == 1: length = self.shape[2]
            else: length = self.shape[3]

            for coord in range(self.around, length - self.around):

                yield functools.partial(self.get_slice, dim, coord)

    def get_all_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Retrieve all slice callables that have segmentation as partials.

        This function collects and returns a list of callable functions that,
        when invoked, yield tuples containing two torch.Tensor objects. These
        callables are specifically related to slices that involve segmentation
        as partials, allowing for efficient processing of tensor data.

        Returns:
            list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]: A list of
            callables that return tuples of two torch.Tensor objects.
        """
        return list(self.yield_all_slice_callables())

    def get_all_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve all slices that have segmentation.

        This function iterates through all slice callables obtained from the
        `get_all_slice_callables` method and executes each callable to retrieve
        the corresponding slices. The result is a list of tuples, where each
        tuple contains two tensors representing the segmented slices.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor]]: A list of tuples,
            each containing two tensors that represent the segmented slices.
        """
        return [i() for i in self.get_all_slice_callables()]


    def yield_all_empty_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices, including empty segmentation ones, as partial
        callables.

        This function iterates over the dimensions of a tensor and yields
        callable functions that can retrieve slices of the tensor. It accounts
        for empty segmentation by checking if the current coordinate is not
        present in the corresponding coordinate list. The yielded callables can
        be used to obtain slices of the tensor at specified dimensions and
        coordinates.

        Yields:
            Callable[[], tuple[torch.Tensor, torch.Tensor]]: A callable that returns a tuple of
            tensors corresponding to the specified slice.
        """
        # pick a dimension
        for dim in (0, 1, 2):

            # get length
            if dim == 0:
                coord_list = self.x
                length = self.shape[1]
            elif dim == 1:
                coord_list = self.y
                length = self.shape[2]
            else:
                coord_list = self.z
                length = self.shape[3]
            for coord in range(self.around, length - self.around):
                if coord not in coord_list: yield functools.partial(self.get_slice, dim, coord)

    def get_all_empty_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Retrieve all slice callables that have segmentation as partials.

        This function calls another method to yield all empty slice callables
        and converts the result into a list. The returned callables are expected
        to return a tuple containing two torch.Tensor objects when invoked.

        Returns:
            list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]: A list of callables
            that return tuples of two torch.Tensor objects.
        """
        return list(self.yield_all_empty_slice_callables())

    def get_all_empry_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve all slices that contain segmentation.

        This function calls a series of callable objects that are designed to
        return empty slices. It collects all the results into a list of tuples,
        where each tuple consists of two torch.Tensor objects. This is useful
        for obtaining a comprehensive view of all empty slices available in the
        current context.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor]]: A list of tuples,
            each containing two torch.Tensor objects representing the
            empty slices with segmentation.
        """
        return [i() for i in self.get_all_empty_slice_callables()]

    def get_non_empty_count(self): return len(self.x) + len(self.y) + len(self.z)

    def get_anyp_random_slice_callables(self):
        """Generate a list of callable functions for random slicing.

        This function calculates the number of callable functions to be
        generated based on the probability of selecting any segment. It computes
        the ratio of the probability of selecting any segment to the probability
        of selecting a specific segment, and then returns a list of callables
        that can be used to perform random slicing operations.

        Returns:
            list: A list of callable functions for random slicing.
        """

        seg_prob = 1 - self.any_prob
        any_to_seg_ratio = self.any_prob / seg_prob
        return [self.get_random_slice for i in range(int(self.get_non_empty_count() * any_to_seg_ratio))]
