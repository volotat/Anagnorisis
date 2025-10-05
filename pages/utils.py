import math
import datetime
from dateutil.relativedelta import relativedelta

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


import time

###########################################
# Sorting Progress Callback

class SortingProgressCallback:
    def __init__(self, show_status_function=None, operation_name="Sorting"):
        self.last_shown_time = 0
        self.start_time = time.time()
        self.show_status_function = show_status_function
        self.operation_name = operation_name

    def __call__(self, num_processed, num_total):
        current_time = time.time()
        if current_time - self.last_shown_time >= 1:  
            # Calculate the percentage of processed items
            percent = (num_processed / num_total) * 100

            # Show the status
            self.show_status_function(f"{self.operation_name} {percent:.2f}% ({num_processed}/{num_total} files)")
            self.last_shown_time = current_time


###########################################
# Embedding Gathering Progress Callback

class EmbeddingGatheringCallback:
    def __init__(self, show_status_function=None, name=""):
        self.last_shown_time = 0
        self.start_time = time.time()
        self.show_status_function = show_status_function
        self.name = name

    def __call__(self, num_extracted, num_total):
        current_time = time.time()
        if current_time - self.last_shown_time >= 1:
            # Calculate the percentage of processed files
            percent = (num_extracted / num_total) * 100

            # Show the status
            self.show_status_function(f"Extracted {self.name} embeddings for {num_extracted}/{num_total} ({percent:.2f}%) files.")
            self.last_shown_time = current_time