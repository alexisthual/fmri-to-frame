import collections
import datetime
import logging
import time
import typing as tp

import numpy as np
import PIL
import torch
import torchvision.transforms as T
from PIL import Image
from PIL.Image import Image as pil_image_type
from submitit.core import core


def convert_to_PIL(images: tp.List) -> tp.List:
    """
    Args
    - a list of images (either as paths, or numpy arrays, or PIL images)
    Return
    - a list of corresponding PIL images
    """
    pil_images = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        elif isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy().astype(np.uint8))
        elif isinstance(img, pil_image_type):
            pass
        else:
            raise ValueError(f"{img} has type {type(img)} which is not supported")
        pil_images.append(img)
    return pil_images


def regularize_image(x):
    BICUBIC = PIL.Image.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = T.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = T.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = T.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, "Unknown image type"

    assert (x.shape[1] == 512) & (x.shape[2] == 512), "Wrong image size"
    return x


def get_logger(name="rich"):
    """Return logging logger."""
    logger = logging.getLogger(name)
    return logger


def submitit_monitoring_logging(
    monitoring_start_time: float,
    n_jobs: int,
    state_jobs: tp.Dict[str, tp.Set[int]],
    logger=None,
):
    """Log submitit jobs status."""
    if logger is None:
        logger = get_logger()

    run_time = time.time() - monitoring_start_time
    n_chars = len(str(n_jobs))

    # Clear last 3 lines from previous logging
    # for _ in range(3):
    #     print("\033[F\033[K", end="")
    # s = "\033[F\033[K" * 3

    logger.info(
        # f"{s}"
        "Jobs running, failed, done, total:\t"
        f"{len(state_jobs['RUNNING']):0{n_chars}}\t"
        f"{len(state_jobs['FAILED']):0{n_chars}}\t"
        f"{len(state_jobs['DONE']):0{n_chars}}\t"
        f"{n_jobs}\t"
        "Duration: "
        f"{str(datetime.timedelta(seconds=int(run_time)))}\n"
        f"RUNNING: {state_jobs['RUNNING']}\n"
        f"FAILED: {state_jobs['FAILED']}\n"
        f"DONE: {state_jobs['DONE']}"
    )


def monitor_jobs(
    jobs: tp.Sequence[core.Job[core.R]],
    poll_frequency: float = 60,
    test_mode: bool = False,
    custom_logging: tp.Callable = submitit_monitoring_logging,
    logger=None,
) -> None:
    """Monitor given jobs continuously until they are all done or failed.

    Taken from submitit and adapted to use a custom logging logger.

    Parameters
    ----------
    jobs: List[Jobs]
        A list of jobs to monitor
    poll_frequency: int
        The time (in seconds) between two refreshes of the monitoring.
        Can't be inferior to 30s.
    test_mode: bool
        If in test mode, we do not check the length of poll_frequency
    """

    # Handle logger
    if logger is None:
        logger = get_logger()

    # # Remove all existing handlers
    # for handler in logger.handlers:
    #     logger.removeHandler(handler)

    # # Create a new handler and set the custom formatter
    # handler = logging.StreamHandler()
    # handler.setFormatter(CustomFormatter())
    # logger.addHandler(handler)

    if not test_mode:
        assert (
            poll_frequency >= 30
        ), "You can't refresh too often (>= 30s) to avoid overloading squeue"

    n_jobs = len(jobs)
    if n_jobs == 0:
        logger.info("There are no jobs to monitor")
        return

    # Add a new line to avoid overwriting the last line of the logger
    print("\n\n\n", sep="")

    job_arrays = ", ".join(
        sorted(set(str(job.job_id).split("_", 1)[0] for job in jobs))
    )
    logger.info(f"Monitoring {n_jobs} jobs from job arrays {job_arrays}")

    monitoring_start_time = time.time()
    running_start_time = None
    while True:
        if not test_mode:
            jobs[0].get_info(mode="force")  # Force update once to sync the state
        state_jobs = collections.defaultdict(set)
        for i, job in enumerate(jobs):
            state_jobs[job.state.upper()].add(i)
            if job.done():
                state_jobs["DONE"].add(i)

        if running_start_time is None and (
            len(state_jobs["RUNNING"]) > 0 or len(state_jobs["DONE"]) > 0
        ):
            running_start_time = time.time()

        failed_job_indices = sorted(state_jobs["FAILED"])
        if len(state_jobs["DONE"]) == len(jobs):
            logger.info(
                "All jobs finished, duration since first run"
                f" {str(datetime.timedelta(seconds=int(time.time() - running_start_time)))},"
                " total duration"
                f" {str(datetime.timedelta(seconds=int(time.time() - monitoring_start_time)))},"
                f" jobs {failed_job_indices} failed"
            )
            break

        custom_logging(monitoring_start_time, n_jobs, state_jobs, logger=logger)
        time.sleep(poll_frequency)
