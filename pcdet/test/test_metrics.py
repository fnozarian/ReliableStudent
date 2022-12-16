print("start test_metrics.py")
import unittest
import sys
import os
import numpy as np

import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
import torch.distributed as dist

sys.path.append('/tmp/OpenPCDet/pcdet/utils')
from stats_utils import KITTIEvalMetrics
from stats_utils import PredQualityMetrics

# from https://github.com/Lightning-AI/metrics/blob/9b19a922487e295810bf5e22a587727964cc8718/tests/unittests/bases/test_ddp.py
import pytest
from torch import tensor
from torchmetrics import Metric

MAX_PORT = 8100
START_PORT = 8088
CURRENT_PORT = START_PORT


# gloo has the advantage that tests can be run
# on a local machine (no need for distributed multi-gpu env).
def setup_ddp_gloo(rank, world_size):
    """Setup ddp environment."""
    global CURRENT_PORT

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(CURRENT_PORT)

    CURRENT_PORT += 1
    if CURRENT_PORT > MAX_PORT:
        CURRENT_PORT = START_PORT

    if torch.distributed.is_available():
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)



class DummyMetric(Metric):
    name = "Dummy"
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("x", tensor(0.0), dist_reduce_fx="sum")

    def update(self):
        pass

    def compute(self):
        pass


class DummyListMetric(Metric):
    name = "DummyList"
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("x", [], dist_reduce_fx="cat")

    def update(self, x=torch.tensor(1)):
        self.x.append(x)

    def compute(self):
        return self.x


class DummyMetricSum(DummyMetric):
    def update(self, x):
        self.x += x

    def compute(self):
        return self.x


def _test_sync_on_compute_tensor_state(rank, worldsize, sync_on_compute):
    setup_ddp_gloo(rank, worldsize)
    dummy = DummyMetricSum(sync_on_compute=sync_on_compute)
    dummy.update(tensor(rank + 1))
    val = dummy.compute()
    if sync_on_compute:
        assert val == 3
    else:
        assert val == rank + 1


def _test_sync_on_compute_list_state(rank, worldsize, sync_on_compute):
    setup_ddp_gloo(rank, worldsize)
    dummy = DummyListMetric(sync_on_compute=sync_on_compute)
    dummy.update(tensor(rank + 1))
    val = dummy.compute()
    if sync_on_compute:
        assert torch.allclose(val, tensor([1, 2]))
    else:
        assert val == [tensor(rank + 1)]


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize("sync_on_compute", [True, False])
@pytest.mark.parametrize("test_func", [_test_sync_on_compute_list_state, _test_sync_on_compute_tensor_state])
def test_sync_on_compute(sync_on_compute, test_func):
    """Test that syncronization of states can be enabled and disabled for compute."""
    torch.multiprocessing.spawn(test_func, args=(2, sync_on_compute), nprocs=2)

# Uncomment when testing on a single machine/gpu
def _gloo_distributed_mean_iou(rank, worldsize):
    setup_ddp_gloo(rank, worldsize)
    batch_size = 2
    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    model = MyTestModule(dataset)
    dataloader_iter = iter(dataloader)
    res = {}
    while 'pred_ious' not in res:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        batch = [[a for a in arg] for arg in batch]
        res = model(*batch)

    calculated = res['pred_ious']
    expected = dataset.pseudo_ious
    print(f"calculated: {calculated}")
    print(f"expected: {expected}")
    # print(f"is sync_on_compute: {model.map_metric.sync_on_compute}")
    for k, v in expected.items():
        assert np.allclose(calculated[k], v, atol=1e-4)


@pytest.mark.parametrize("num_procs", [2])
def test_gloo_distributed_mean_iou(num_procs):
    """Test that syncronization of states can be enabled and disabled for compute."""
    torch.multiprocessing.spawn(_gloo_distributed_mean_iou, args=(num_procs,), nprocs=num_procs)


class MyDataset(torch_data.Dataset):
    def __init__(self):
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.preds_dataset = [torch.tensor([[25.2296, -4.1550, -0.6772, 4.0161, 1.5722, 1.7488, -2.7502, 1.0000],
                                            [7.7078, 6.1993, -0.9011, 3.9870, 1.5385, 1.5205, -6.0359, 1.0000],
                                            [19.2541, 9.7261, -0.8319, 3.8077, 1.5632, 1.4834, -5.8402, 1.0000],
                                            [2.1382, 4.1422, -0.9770, 3.8312, 1.6293, 1.4491, -5.9839, 1.0000],
                                            [23.5270, 11.3895, -0.7414, 3.5211, 1.5604, 1.5818, -5.9583, 1.0000],
                                            [53.1898, 8.3655, -0.7060, 3.8584, 1.5285, 1.4132, -2.6382, 1.0000],
                                            [44.5012, 18.5638, -0.7170, 3.6333, 1.5717, 1.4514, -5.8257, 1.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                              torch.tensor([[24.6630, -3.0154, -0.7441, 4.1083, 1.6835, 1.8868, -3.0896, 1.0000],
                                            [8.0994, 6.8529, -0.5462, 0.5872, 0.6771, 1.9583, -3.9162, 2.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                              torch.tensor([[25.2296, -4.1550, -0.6772, 4.0161, 1.5722, 1.7488, -2.7502, 1.0000],
                                            [7.7078, 6.1993, -0.9011, 3.9870, 1.5385, 1.5205, -6.0359, 1.0000],
                                            [19.2541, 9.7261, -0.8319, 3.8077, 1.5632, 1.4834, -5.8402, 1.0000],
                                            [2.1382, 4.1422, -0.9770, 3.8312, 1.6293, 1.4491, -5.9839, 1.0000],
                                            [23.5270, 11.3895, -0.7414, 3.5211, 1.5604, 1.5818, -5.9583, 1.0000],
                                            [53.1898, 8.3655, -0.7060, 3.8584, 1.5285, 1.4132, -2.6382, 1.0000],
                                            [44.5012, 18.5638, -0.7170, 3.6333, 1.5717, 1.4514, -5.8257, 1.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                              torch.tensor([[24.6630, -3.0154, -0.7441, 4.1083, 1.6835, 1.8868, -3.0896, 1.0000],
                                            [8.0994, 6.8529, -0.5462, 0.5872, 0.6771, 1.9583, -3.9162, 2.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
                              ]
        self.targets_dataset = [torch.tensor([[2.4623, 4.0398, -0.9571, 3.6328, 1.4648, 1.3672, 0.2663, 1.0000],
                                              [7.7829, 6.2907, -0.8967, 4.3066, 1.6016, 1.4844, 0.2963, 1.0000],
                                              [25.4319, -4.1348, -0.5923, 4.0137, 1.6406, 1.8066, -2.7269, 1.0000],
                                              [53.3162, 8.5708, -0.7615, 3.7500, 1.5430, 1.4551, -2.6869, 1.0000],
                                              [19.0131, 9.6793, -0.7809, 3.4375, 1.5332, 1.5039, 0.4463, 1.0000],
                                              [23.5485, 11.4648, -0.8890, 3.4863, 1.5918, 1.4648, 0.3363, 1.0000],
                                              [29.0273, 13.3456, -0.9067, 3.5547, 1.5527, 1.4355, 0.3763, 1.0000],
                                              [44.3464, 18.6039, -0.8752, 3.2031, 1.5430, 1.4062, 0.3563, 1.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                                torch.tensor([[8.0372, 6.8608, -0.4496, 1.0894, 0.4400, 1.9903, 1.8125, 2.0000],
                                              [9.0257, 7.2973, -0.5628, 0.9428, 0.5238, 1.9589, 1.9925, 2.0000],
                                              [24.6333, -3.0353, -0.7210, 4.0121, 1.6446, 1.9484, -3.0607, 1.0000],
                                              [32.5008, 5.6985, -0.4304, 0.9952, 0.5552, 1.8646, 1.4525, 2.0000],
                                              [12.8716, -7.2570, -1.0778, 0.4295, 0.5971, 1.6132, -3.0307, 2.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                                torch.tensor([[2.4623, 4.0398, -0.9571, 3.6328, 1.4648, 1.3672, 0.2663, 1.0000],
                                              [7.7829, 6.2907, -0.8967, 4.3066, 1.6016, 1.4844, 0.2963, 1.0000],
                                              [25.4319, -4.1348, -0.5923, 4.0137, 1.6406, 1.8066, -2.7269, 1.0000],
                                              [53.3162, 8.5708, -0.7615, 3.7500, 1.5430, 1.4551, -2.6869, 1.0000],
                                              [19.0131, 9.6793, -0.7809, 3.4375, 1.5332, 1.5039, 0.4463, 1.0000],
                                              [23.5485, 11.4648, -0.8890, 3.4863, 1.5918, 1.4648, 0.3363, 1.0000],
                                              [29.0273, 13.3456, -0.9067, 3.5547, 1.5527, 1.4355, 0.3763, 1.0000],
                                              [44.3464, 18.6039, -0.8752, 3.2031, 1.5430, 1.4062, 0.3563, 1.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                                torch.tensor([[8.0372, 6.8608, -0.4496, 1.0894, 0.4400, 1.9903, 1.8125, 2.0000],
                                              [9.0257, 7.2973, -0.5628, 0.9428, 0.5238, 1.9589, 1.9925, 2.0000],
                                              [24.6333, -3.0353, -0.7210, 4.0121, 1.6446, 1.9484, -3.0607, 1.0000],
                                              [32.5008, 5.6985, -0.4304, 0.9952, 0.5552, 1.8646, 1.4525, 2.0000],
                                              [12.8716, -7.2570, -1.0778, 0.4295, 0.5971, 1.6132, -3.0307, 2.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
                                ]
        self.pred_scores_dataset = [torch.tensor([[1.0000],
                                                  [1.0000],
                                                  [1.0000],
                                                  [0.9999],
                                                  [0.9998],
                                                  [0.9716],
                                                  [0.9398],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000]]),
                                    torch.tensor([[1.0000],
                                                  [0.9199],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000]]),
                                    torch.tensor([[1.0000],
                                                  [1.0000],
                                                  [1.0000],
                                                  [0.9999],
                                                  [0.9998],
                                                  [0.9716],
                                                  [0.9398],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000]]),
                                    torch.tensor([[1.0000],
                                                  [0.9199],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000],
                                                  [0.0000]])
                                    ]
        self.pred_sem_scores_dataset = [torch.tensor([[0.7000],
                                                      [0.6000],
                                                      [0.5000],
                                                      [0.4000],
                                                      [0.3000],
                                                      [0.2000],
                                                      [0.9000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000]]),
                                        torch.tensor([[0.9000],
                                                      [0.7000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000]]),
                                        torch.tensor([[0.7000],
                                                      [0.6000],
                                                      [0.5000],
                                                      [0.4000],
                                                      [0.3000],
                                                      [0.2000],
                                                      [0.9000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000]]),
                                        torch.tensor([[0.9000],
                                                      [0.7000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000],
                                                      [0.0000]])
                                        ]
        # self.pseudo_ious = [0.7289696335792542, 0.6714359521865845]
        self.pseudo_ious = {'Car': 0.8187686204910278, 'Pedestrian': 0.4343862235546112, 'cls_agnostic': 0.7002184987068176}

    def __getitem__(self, index):
        return (self.preds_dataset[index].cuda(),
                self.targets_dataset[index].cuda(),
                self.pred_scores_dataset[index].cuda().squeeze(),
                self.pred_sem_scores_dataset[index].cuda().squeeze())

    def __len__(self):
        return len(self.preds_dataset)


class MyTestModule(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.map_metric = PredQualityMetrics(dataset=dataset, reset_state_interval=4)

    def forward(self, pseudo_boxes_list, ori_boxes_list, pseudo_scores, pseudo_sem_scores):
        self.map_metric.update(pseudo_boxes_list, ori_boxes_list, pseudo_scores, pseudo_sem_scores)
        results = self.map_metric.compute()
        return results


# def setup_ddp_nccl():
#     # assuming master_addr, master_port, rank and world_size
#     # are already set in env by Slurm.
#     proc_id = int(os.environ['SLURM_PROCID'])
#     num_gpus = torch.cuda.device_count()
#     rank = int(os.environ['RANK'])
#     torch.cuda.set_device(proc_id % num_gpus)
#     dist.init_process_group(backend='nccl')
#     total_gpus = dist.get_world_size()
#
#     return total_gpus, rank


# class TestDistributedMetrics(unittest.TestCase):
#     def test_torchmetrics_distributed_mean_iou(self):
#         batch_size = 2
#         setup_ddp_nccl()
#         dataset = MyDataset()
#         sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
#         model = MyTestModule()
#         model.cuda()
#         dataloader_iter = iter(dataloader)
#         batch = next(dataloader_iter)
#         print("call model forward")
#         res = model(*batch)
#         calculated = res['pseudo_ious']
#         expected = np.mean(dataset.pseudo_ious)
#         print(f"calculated: {calculated}")
#         print(f"expected: {expected}")
#         self.assertAlmostEqual(calculated, expected, delta=1e-4)


# if __name__ == "__main__":
#     # run_tests()
#     unittest.main()