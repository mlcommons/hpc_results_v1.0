# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import ctypes
import logging.config
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.init as init
import torch.utils.collect_env
from mlperf_logging.mllog import constants
from mlperf_logging import mllog
import pandas as pd
from collections import OrderedDict

#comm wrapper
from utils import comm

class mlperf_logger(object):

    def __init__(self, filename, benchmark, organization, platform):
        self.mllogger = mllog.get_mllogger()
        self.comm_rank = comm.get_rank()
        self.comm_size = comm.get_size()
        self.constants = constants

        # create logging dir if it does not exist
        logdir = os.path.dirname(filename)
        if self.comm_rank == 0:
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # create config
        mllog.config(filename = filename)
        self.mllogger.logger.propagate = False
        self.log_event(key = constants.SUBMISSION_BENCHMARK,
                       value = benchmark)

        self.log_event(key = constants.SUBMISSION_ORG,
                       value = organization)
        
        self.log_event(key = constants.SUBMISSION_DIVISION,
                       value = 'closed')

        self.log_event(key = constants.SUBMISSION_STATUS,
                       value = 'onprem')

        self.log_event(key = constants.SUBMISSION_PLATFORM,
                       value = platform)
        

    def log_start(self, *args, **kwargs):
        self._log_print(self.mllogger.start, *args, **kwargs)
        
    def log_end(self, *args, **kwargs):
        self._log_print(self.mllogger.end, *args, **kwargs)
        
    def log_event(self, *args, **kwargs):
        self._log_print(self.mllogger.event, *args, **kwargs)

    def _log_print(self, logger, *args, **kwargs):
        """
        Wrapper for MLPerf compliance logging calls.
        All arguments but 'sync' and 'log_all_ranks' are passed to
        mlperf_logging.mllog.
        If 'sync' is set to True then the wrapper will synchronize all distributed
        workers. 'sync' should be set to True for all compliance tags that require
        accurate timing (RUN_START, RUN_STOP etc.)
        If 'log_all_ranks' is set to True then all distributed workers will print
        logging message, if set to False then only worker with rank=0 will print
        the message.
        """
        if kwargs.pop('sync', False):
            self.barrier()
        if 'stack_offset' not in kwargs:
            kwargs['stack_offset'] = 3
        if 'value' not in kwargs:
            kwargs['value'] = None

        if kwargs.pop('log_all_ranks', False):
            log = True
        else:
            log = (self.comm_rank == 0)

        if log:
            logger(*args, **kwargs)

    def barrier(self):
        """
        Works as a temporary distributed barrier, currently pytorch
        doesn't implement barrier for NCCL backend.
        Calls all_reduce on dummy tensor and synchronizes with GPU.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
                    

def save_config(pargs):
    config_file = os.path.normpath(os.path.join(pargs.output_dir, "logs", pargs.run_tag + "_config.pkl"))
    print('Writing config via pickle to {:s}'.format(config_file))
    with open(config_file, 'wb') as f:
        pickle.dump(pargs, f)
    env_file = os.path.normpath(os.path.join(pargs.output_dir, "logs", pargs.run_tag + "_env.pkl"))
    print('Writing environment variables via pickle to {:s}'.format(env_file))
    with open(env_file, 'wb') as f:
        pickle.dump({k: v for k, v in os.environ.items() if
                     k.startswith('SLURM_') or
                     k.startswith('NCCL_') or
                     k.startswith('PYTORCH_') or
                     k.startswith('PT_')}, f)


class history_logger(object):
    """Recording time, training loss and training/validation score of each epoch"""
    def __init__(self, history_log_file):
        self.comm_rank = comm.get_rank()
        if self.comm_rank == 0:
            self.history_filename = history_log_file
            self.history_file = open(history_log_file, 'w')
        self.epoch_log = OrderedDict(epoch=-1, lr=-1,
                                     time=-1, train_loss=-1, train_accuracy=-1,
                                     val_time=-1, val_loss=-1, val_accuracy=-1)
        if self.comm_rank == 0:
            pd.DataFrame(self.epoch_log, index=[0]).drop(0).to_csv(self.history_file, index=False)

    def on_run_start(self):
        self.run_start = time.time()

    def on_epoch_begin(self, epoch):
        self.epoch_log = OrderedDict(epoch=-1, lr=-1,
                                     time=-1, train_loss=-1, train_accuracy=-1,
                                     val_time=-1, val_loss=-1, val_accuracy=-1)
        self.epoch_start = time.time()
        self.epoch_log['epoch'] = epoch

    def on_validation_begin(self, epoch):
        self.validation_start = time.time()

    def on_validation_end(self, epoch, **val_vars):
        validation_time = time.time() - self.validation_start
        self.epoch_log['val_time'] = validation_time
        self.epoch_log.update(val_vars)

    def on_epoch_end(self, epoch, **train_vars):
        epoch_time = time.time() - self.epoch_start
        self.epoch_log['time'] = epoch_time
        self.epoch_log.update(train_vars)
        if self.comm_rank == 0:
            pd.DataFrame(self.epoch_log, index=[epoch]).to_csv(self.history_file, index=False, header=False, mode='a')

    def on_run_stop(self):
        self.mlperf_run_time = time.time() - self.run_start

    def close(self, print_summary=False, train_samples=None, val_samples=None):
        if self.comm_rank == 0:
            self.history_file.close()
            if not print_summary:
                return

            history = pd.read_csv(self.history_filename)
            if 'val_accuracy' in history.keys():
                best = history.val_accuracy.idxmax()
                print('Best result:')
                for key in history.keys():
                    print('  {:s}: {:g}'.format(key, history[key].loc[best]))

            # Compute relative distribution
            compute_time = history.time.sum()
            compute_share = compute_time / self.mlperf_run_time
            eval_compute_share = history.val_time.sum() / history.time.sum()
            train_compute_share = 1 - eval_compute_share
            eval_share = compute_share * eval_compute_share
            train_share = compute_share * train_compute_share

            print('Runtime statistics')
            print('  total run time:              {:.3f} sec ({:.2f}% train, {:.2f}% eval, {:.2f}% other)'.format(
                         self.mlperf_run_time, 100 * train_share, 100 * eval_share, 100 * (1 - compute_share)))
            print('  total compute time (epochs): {:.3f} sec ({:.2f}% train, {:.2f}% eval)'.format(
                         compute_time, 100 * train_compute_share, 100 * eval_compute_share))
            eval_epoch_share = history.val_time / history.time
            train_epoch_share = 1 - eval_epoch_share
            print(
                '  mean epoch time:             {:.3f} sec +- {:.2f}% ({:.2f}% +- {:.2f}% train, {:.2f}% +- {:.2f}% eval)'.format(
                history.time.mean(), 100 * history.time.std() / history.time.mean(),
                100 * train_epoch_share.mean(), 100 * train_epoch_share.std(),
                100 * eval_epoch_share.mean(), 100 * eval_epoch_share.std())
                )

            if train_samples is not None and val_samples is not None:
                def first_later_epochs_throughput(n_samples, epoch_times):
                    return n_samples / epoch_times[0], (n_samples / epoch_times[1:]).mean()

                first_epoch_comb_throughput, later_epochs_comb_throughput = \
                    first_later_epochs_throughput(train_samples + val_samples, history.time)
                first_epoch_train_throughput, later_epochs_train_throughput = \
                    first_later_epochs_throughput(train_samples, history.time - history.val_time)
                first_epoch_val_throughput, later_epochs_val_throughput = \
                    first_later_epochs_throughput(val_samples, history.val_time)

                def single_rank_throughput(system_throughput):
                    return system_throughput / comm.get_size()

                print('Sample throughput (first vs later epochs)')
                print('                '
                      'comb [samples/sec]:     '
                      'train [samples/sec]:    '
                      'eval [samples/sec]:     ')
                print('  first epoch:  '
                      '{:.3f} ({:.3f} per rank), {:.3f} ({:.3f} per rank), {:.3f} ({:.3f} per rank)  '.format(
                      first_epoch_comb_throughput, single_rank_throughput(first_epoch_comb_throughput),
                      first_epoch_train_throughput, single_rank_throughput(first_epoch_train_throughput),
                      first_epoch_val_throughput, single_rank_throughput(first_epoch_val_throughput))
                      )
                print('  later epochs: '
                      '{:.3f} ({:.3f} per rank), {:.3f} ({:.3f} per rank), {:.3f} ({:.3f} per rank)  '.format(
                      later_epochs_comb_throughput, single_rank_throughput(later_epochs_comb_throughput),
                      later_epochs_train_throughput, single_rank_throughput(later_epochs_train_throughput),
                      later_epochs_val_throughput, single_rank_throughput(later_epochs_val_throughput))
                      )
                print('  speedup f->l: '
                      '{:.3f}x                , {:.3f}x                , {:.3f}x                  '.format(
                      later_epochs_comb_throughput / first_epoch_comb_throughput,
                      later_epochs_train_throughput / first_epoch_train_throughput,
                      later_epochs_val_throughput / first_epoch_val_throughput)
                      )
