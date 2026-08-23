import math
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def convert_length(length_seconds):
    minutes, seconds = divmod(length_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    seconds = round(seconds)
    minutes = round(minutes)
    hours = round(hours)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def time_difference(timestamp1, timestamp2):
    dt1 = datetime.datetime.fromtimestamp(timestamp1)
    dt2 = datetime.datetime.fromtimestamp(timestamp2)
    total_seconds = (dt2 - dt1).total_seconds()
    diff = relativedelta(dt2, dt1)

    if total_seconds < 60:
        return "Just now"

    readable_form = []
    if diff.years > 0:
        readable_form.append(f"{diff.years} years")
    if diff.months > 0:
        readable_form.append(f"{diff.months} months")
    if diff.days > 0:
        readable_form.append(f"{diff.days} days")
    if diff.hours > 0:
        readable_form.append(f"{diff.hours} hours")
    if diff.minutes > 0:
        readable_form.append(f"{diff.minutes} minutes")

    return " ".join(readable_form) + " ago"

# The progress adapters moved to anagnorisis_core.progress, which is where the
# engine reports from. Re-exported here so the application's modules keep their
# existing import; new code should take them from the core.
from anagnorisis_core.progress import (  # noqa: E402,F401  (compatibility re-export)
    ArbitraryProgressCallback, EmbeddingGatheringCallback, SortingProgressCallback,
)
