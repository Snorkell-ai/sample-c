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
    """Recursively moves x to device, if possible. Note that this can be very slow.

    Args:
        x (_type_): _description_
        device (Optional[torch.device]): _description_

    Returns:
        _type_: _description_
    """
    if device is None: return x
    if isinstance(x, torch.Tensor): return x.to(device)
    elif isinstance(x, (list, tuple)): return [ensure_device(i, device) for i in x]
    else: return x

def ensure_detach(x) -> Any:
    """Recursively detaches x, if possible. Note that this can be very slow.

    Args:
        x (_type_): _description_

    Returns:
        Any: _description_
    """
    if isinstance(x, torch.Tensor): return x.detach()
    elif isinstance(x, (list, tuple)): return [ensure_detach(i) for i in x]
    else: return x

def ensure_cpu(x) -> Any:
    """Recursively moves x to cpu, if possible. Note that this can be very slow.

    Args:
        x (_type_): _description_

    Returns:
        Any: _description_
    """
    if isinstance(x, torch.Tensor): return x.cpu()
    elif isinstance(x, (list, tuple)): return [ensure_cpu(i) for i in x]
    else: return x

def ensure_detach_cpu(x) -> Any:
    """Recursively detaches x and moves it to cpu, if possible. Note that this can be very slow.

    Args:
        x (_type_): _description_

    Returns:
        Any: _description_
    """
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    elif isinstance(x, (list, tuple)): return [ensure_detach_cpu(i) for i in x]
    else: return x

def ensure_float(x) -> Any:
    """    Save the processed files map to a JSON file.

    Function parameters should be documented in the ``Args`` section. The
    name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    Args:
        dictionary (dict): The processed files map.

    Returns:
        bool: True if successful, False otherwise.
        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.
        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.
        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::
            
            {
            'param1': param1,
            'param2': param2
            }
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
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = self.original_requires_grads[i]

        self.frozen = False


def is_container(mod:torch.nn.Module):
    """Returns True if the module is a container"""
    if len(list(mod.children())) == 0: return False # all containers have chilren
    if len(list(mod.parameters(False))) == 0 and len(list(mod.buffers(False))) == 0: return True # containers don't do anything themselves so they can't have parameters or buffers
    return False # has children, but has params or buffers

def param_count(module:torch.nn.Module): return sum(p.numel() for p in module.parameters())
def buffer_count(module:torch.nn.Module): return sum(b.numel() for b in module.buffers())


def _summary_hook(path:str, module:torch.nn.Module, input:tuple[torch.Tensor], output: torch.Tensor):#pylint:disable=W0622
    input_info = '; '.join([(str(tuple(i.size())) if hasattr(i, "size") else str(i)[:100]) for i in input])
    print(
        f"{path:<45}{type_str(module):<45}{input_info:<25}{str(tuple(output.size())):<25}{param_count(module):<10}{buffer_count(module):<10}"
    )

def _register_summary_hooks(hooks:list, name:str, path:str, module:torch.nn.Module):
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
    "Print a summary table of `module`."
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
        if model is None: model = self.model
        self.state_dict = copy_state_dict(model.state_dict()) # type:ignore

    def restore(self, model:Optional[torch.nn.Module] = None):
        if model is None: model = self.model
        model.load_state_dict(copy_state_dict(self.state_dict)) # type:ignore
        return model


def get_lr(optimizer:torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer:torch.optim.Optimizer, lr:int|float):
    for g in optimizer.param_groups:
        g['lr'] = lr

def change_lr(optimizer:torch.optim.Optimizer, fn:Callable):
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
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return t


def to_binary(t:torch.Tensor, threshold:float = 0.5):
    return torch.where(t > threshold, 1, 0)


def center_of_mass(feature:torch.Tensor):
    '''
    https://github.com/tym002/tensorflow_compute_center_of_mass/blob/main/compute_center_mass.py

    COM computes the center of mass of the input 4D or 5D image
    To use COM in a tensorflow model, use layers.Lambda
    Arguments:
        feature: input image of 5D tensor with format [batch,x,y,z,channel]
                    or 4D tensor with format [batch,x,y,channel]
        nx,ny,nz: dimensions of the input image, if using 4D tensor, nz = None
    '''
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
    """
    Erodes a 3D binary tensor.
    """
    if n > 1: tensor = binary_erode3d(tensor, n-1)
    kernel = torch.tensor([[[[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]]], dtype=torch.int64)
    convolved = torch.nn.functional.conv3d(input = tensor.unsqueeze(0), weight = kernel, padding=1) # pylint:disable=E1102
    return torch.where(convolved==7, 1, 0)[0]



def area_around(tensor:torch.Tensor, coord, size) -> torch.Tensor:
    """Returns a tensor of `size` size around `coord`"""
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
    if mask.ndim == 3:
        return torch.nn.functional.one_hot(mask.to(torch.int64), num_classes).permute(3, 0, 1, 2).to(torch.float32) # pylint:disable=E1102 #type:ignore
    elif mask.ndim == 2:
        return torch.nn.functional.one_hot(mask.to(torch.int64), num_classes).permute(2, 0, 1).to(torch.float32) # pylint:disable=E1102 #type:ignore
    else: raise NotImplementedError(f'one_hot_mask: mask.ndim = {mask.ndim}')


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def replace_layers(model:torch.nn.Module, old:type, new:torch.nn.Module):
    """https://www.kaggle.com/code/ankursingh12/why-use-setattr-to-replace-pytorch-layers"""
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)

def replace_conv(model:torch.nn.Module, old:type, new:type):
    """Bias always True!!!"""
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_conv(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new(module.in_channels, module.out_channels, module.kernel_size,
                                  module.stride, module.padding, module.dilation, module.groups))

def replace_conv_transpose(model:torch.nn.Module, old:type, new:type):
    """Bias always True!!!"""
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_conv(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new(module.in_channels, module.out_channels, module.kernel_size,
                                  module.stride, module.padding, module.output_padding, module.groups, True, module.dilation))


def unonehot(mask: torch.Tensor, batch = False) -> torch.Tensor:
    if batch: return torch.argmax(mask, dim=1)
    return torch.argmax(mask, dim=0)


def preds_batch_to_onehot(preds:torch.Tensor):
    return one_hot_mask(preds.argmax(1), preds.shape[1]).swapaxes(0,1)


def angle(a, b, dim=-1):
    """https://github.com/pytorch/pytorch/issues/59194"""
    a_norm = a.norm(dim=dim, keepdim=True)
    b_norm = b.norm(dim=dim, keepdim=True)
    return 2 * torch.atan2(
        (a * b_norm - a_norm * b).norm(dim=dim),
        (a * b_norm + a_norm * b).norm(dim=dim)
    )

@contextmanager
def seeded_rng(seed:Optional[Any]=0):
    """Context manager, sets seed to torch,numpy and random. If seed is None, does nothing."""
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
    """
    ```py
    DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,)
    ```
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
    with seeded_rng(seed):
        return torch.randperm(n, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)

def stepchunk(vec:torch.Tensor|np.ndarray, chunks:int, maxlength:Optional[int]=None):
    maxlength = maxlength or vec.shape[0]
    return [vec[i : i+maxlength : chunks] for i in range(chunks)]

class ConcatZeroChannelsToDataloader:
    """Wraps dataloader and adds zero channels to the end, useful when model accepts more channels than images have"""
    def __init__(self, dataloader, resulting_channels):
        self.dataloader = dataloader
        self.resulting_channels=resulting_channels
    def __len__(self): return len(self.dataloader)
    def __iter__(self):
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
        for inputs, targets in self.dataloader:
            yield self.transforms(inputs), targets

def map_to_base_np(number:int, base):
    """
    Convert an integer into a list of digits of that integer in a given base.

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
    """
    Convert an integer into a list of digits of that integer in a given base.

    Args:
        number (int): The integer to convert.
        base (int): The base to convert the integer to.

    Returns:
        numpy.ndarray: An array of digits representing the input integer in the given base.
    """
    if number == 0: return torch.tensor([0])
    # Convert the input numbers to their digit representation in the given base
    digits = torch.tensor([number])
    base_digits = (digits // base**(torch.arange(int(math.log(number) / math.log(base)), -1, -1))) % base

    return base_digits


def sliding_inference_around_3d(input:torch.Tensor, inferer, size, step, around, nlabels):
    """Input must be a 4D C* or 5D BC* tensor"""
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
        if around is not None: self.around = around
        if len(self.x) > 0 and any_prob is not None: self.any_prob = any_prob

    def __call__(self):
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
        """Get a slice from given `dim` and `coord`"""
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
        """Get a random slice, ignores `any_prob`."""
        # pick a dimension
        dim: Literal[0,1,2] = random.choice([0,1,2])

        # get length
        if dim == 0: length = self.shape[1]
        elif dim == 1: length = self.shape[2]
        else: length = self.shape[3]

        coord = random.randrange(0 + self.around, length - self.around)
        return self.get_slice(dim, coord)

    def yield_all_seg_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices that have segmentation as partials."""
        # pick a dimension
        for dim in (0, 1, 2):

            if dim == 0: coord_list = self.x
            elif dim == 1: coord_list = self.y
            else: coord_list = self.z

            for coord in coord_list:

                yield functools.partial(self.get_slice, dim, coord)

    def get_all_seg_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Get all slices that have segmentation as partials."""
        return list(self.yield_all_seg_slice_callables())

    def get_all_seg_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get all slices that have segmentation."""
        return [i() for i in self.get_all_seg_slice_callables()]

    def yield_all_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices, including empty segmentation ones, as partials."""
        # pick a dimension
        for dim in (0, 1, 2):

            # get length
            if dim == 0: length = self.shape[1]
            elif dim == 1: length = self.shape[2]
            else: length = self.shape[3]

            for coord in range(self.around, length - self.around):

                yield functools.partial(self.get_slice, dim, coord)

    def get_all_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Get all slices that have segmentation as partials."""
        return list(self.yield_all_slice_callables())

    def get_all_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get all slices that have segmentation."""
        return [i() for i in self.get_all_slice_callables()]


    def yield_all_empty_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices, including empty segmentation ones, as partials."""
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
        """Get all slices that have segmentation as partials."""
        return list(self.yield_all_empty_slice_callables())

    def get_all_empry_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get all slices that have segmentation."""
        return [i() for i in self.get_all_empty_slice_callables()]

    def get_non_empty_count(self): return len(self.x) + len(self.y) + len(self.z)

    def get_anyp_random_slice_callables(self):
        seg_prob = 1 - self.any_prob
        any_to_seg_ratio = self.any_prob / seg_prob
        return [self.get_random_slice for i in range(int(self.get_non_empty_count() * any_to_seg_ratio))]
