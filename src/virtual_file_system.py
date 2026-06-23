import urllib.parse
import fs.path
import hashlib

def resolve_base_and_path_from_url(fs_url):
    """
    Parses any PyFilesystem2 URL, opens the base filesystem connection,
    and returns a tuple of (base_url, path_in_fs).
    
    Correctly handles the structural differences between local (osfs://) 
    and remote (webdav://, ftp://, sftp://, etc.) URL parsing, without
    tripping on '#' or '?' characters inside directory or filenames.
    """
    if '://' not in fs_url:
        # Fallback for raw system paths
        return "osfs:///", fs.path.abspath(fs.path.normpath(fs_url))

    protocol, remainder = fs_url.split('://', 1)
    
    # 1. SPECIAL CASE: Local Filesystem (OSFS)
    if protocol == 'osfs':
        base_url = "osfs:///"
        # The remainder is the raw absolute path on disk (e.g., /mnt/media/...)
        # We preserve the entire string, preventing '#' from being treated as a fragment
        path_in_fs = fs.path.abspath(fs.path.normpath(remainder))
        return base_url, path_in_fs
    
    # 2. REMOTE CASE: (webdav://, ftp://, sftp://, etc.)
    # Split on the first single '/' to separate the network location (authority) from the path
    if '/' in remainder:
        authority, raw_path = remainder.split('/', 1)
        path_in_fs = '/' + raw_path
    else:
        authority = remainder
        path_in_fs = '/'
        
    # Reconstruct the base connection URL (ensuring a trailing slash)
    base_url = f"{protocol}://{authority}/"
    path_in_fs = fs.path.abspath(fs.path.normpath(path_in_fs))
    
    return base_url, path_in_fs

def calculate_file_hash(my_fs, path_in_fs, chunk_size=65536, hash_algorithm=hashlib.md5):
    """
    Reads a file chunk-by-chunk from any PyFilesystem2 target
    and returns its hash hex digest.
    """
    hasher = hash_algorithm()
    
    # Open in binary read mode
    with my_fs.open(path_in_fs, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            
    return hasher.hexdigest()

def join_fs_url(base_url: str, relative_path: str) -> str:
    """
    Safely joins a PyFilesystem2 URL with a relative path.
    Preserves protocols and trailing slashes (e.g., osfs:/// or webdav://).
    """
    if '://' in base_url:
        # Split into 'osfs' and '/mnt/media/'
        protocol, path_segment = base_url.split('://', 1)
        
        # Perform the standard path-join only on the directory part
        joined_path = fs.path.join(path_segment, relative_path)
        
        # Reconstruct the URL. Since joined_path is absolute (starts with /),
        # this naturally restores the correct number of slashes (e.g., osfs:///...)
        return f"{protocol}://{joined_path}"
    else:
        # Fallback if it is already a plain local path
        return fs.path.join(base_url, relative_path)