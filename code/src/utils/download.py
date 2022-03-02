# libraries
# check if init file required
import sys
import os
import urllib.request
import tarfile
import zipfile


def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """

    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Limit it because rounding errors may cause it to exceed 100%.
    pct_complete = min(1.0, pct_complete)

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract(
        base_url: str,
        filename: str,
        download_dir: str
    ) -> None:
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a zip file or a tar-ball file.

    Arguments
    --------
        base_url: str
            URL for the zip or tar-ball to download
        filename: str
            Name of file to be downloaded, to be concatenated to base_url
        download_dir: str
            Target download and extraction directory
    """

    # Filepath for saving the file downloaded from the internet.
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print("Download of ", filename, " finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Extraction of ", filename, " done.")
    else:
        print("Data has apparently already been downloaded and unpacked or has an unknow compression format.")