import math

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