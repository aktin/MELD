"""
Utilities for interacting with file paths, YAML files, and downloading files
from URLs.

This module provides helper functions to resolve file paths relative to a
base directory, load and parse YAML files into dictionaries, sanitize and
validate URLs, generate safe filenames, and download files from the web.
"""
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, unquote, quote, urlunsplit
from urllib.request import Request, urlopen

import yaml


def resolve_path(path_from_contract: str, base_dir: str | Path | None = None) -> str:
    """
    Resolve a file system path relative to a base directory.

    This function combines a given relative path with a base directory
    or the parent directory of the script. It then resolves the result
    to a full, absolute path.

    :param path_from_contract: A relative path to be resolved.
    :type path_from_contract: str
    :param base_dir: The base directory to resolve the relative path against.
                     If not provided, the parent directory of the current script 
                     will be used.
    :type base_dir: str | Path | None
    :return: The absolute resolved path as a string.
    :rtype: str
    """
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    return str((root / path_from_contract).resolve())


def load_yaml(path: str) -> dict:
    """
    Loads a YAML file and parses its contents into a dictionary.

    :param path: The file path to a YAML file to be loaded.
    :type path: str
    :return: A dictionary representation of the YAML file's contents.
    :rtype: dict
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ValueError: If the specified file is not a valid YAML file.
    """

    if not Path(path).exists():
        raise FileNotFoundError(f"The file {path} does not exist.")
    if not path.endswith(".yaml") and not path.endswith(".yml"):
        raise ValueError(f"The file {path} is not a YAML file.")

    with open(path, "r") as file:
        contract = yaml.safe_load(file)

    return contract


def construct_image_tag(contract: dict) -> str:
    """
    Constructs a formatted image tag string based on the provided contract
    dictionary.

    Parameters:
    contract (dict): Dictionary containing the 'inference' section with
    the keys 'image_tag' and 'version'.

    Returns:
    str: A formatted image tag string in the format "<image_tag>:<version>".
    """
    return f"{contract['inference']['image_tag']}:{contract['inference']['version']}"


def _sanitize_url(url: str) -> str:
    """
    Sanitizes and normalizes a URL to ensure it is safe and conforms to specific rules.

    Parameters:
        url (str): The URL string to sanitize.

    Returns:
        str: A sanitized and normalized URL string.

    Raises:
        ValueError: If the URL does not use "http" or "https" as its scheme.
        ValueError: If the URL does not include a valid hostname.
    """
    url = url.strip()
    parsed = urlsplit(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")

    if not parsed.netloc:
        raise ValueError("URL must include a hostname")

    # Normalize and safely encode the path/query.
    safe_path = quote(unquote(parsed.path), safe="/:%")
    safe_query = quote(unquote(parsed.query), safe="=&?/:+,%")

    # Drop fragment, e.g. #section
    return urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        safe_path,
        safe_query,
        "",
    ))


def safe_filename_from_url(url: str, default: str = "downloaded_file") -> str:
    """
    Generate a safe filename from a URL.

    This function takes a URL and extracts its path to generate a filename
    that avoids unsafe or problematic characters. If the resulting filename
    is hidden, empty, or invalid, it defaults to a provided string. This is
    useful for saving files from URLs while ensuring compatibility with
    various file systems.

    Parameters:
    url : str
        The URL from which to derive the filename.
    default : str, optional
        The default name to fall back on if the generated filename is invalid
        or empty. Defaults to "downloaded_file".

    Returns:
    str
        A sanitized and safe filename derived from the input URL.
    """
    parsed = urlsplit(url)
    name = Path(unquote(parsed.path)).name or default

    # Remove characters that are unsafe/problematic in filenames.
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)

    # Avoid hidden or empty filenames.
    name = name.strip("._") or default

    return name


def download_file(url: str, output_dir: str = ".") -> Path:
    """
    Downloads a file from a given URL to a specified directory.

    Arguments:
    url: str
        The URL of the file to be downloaded.
    output_dir: str, optional
        The directory where the downloaded file will be saved. Defaults to the
        current working directory.

    Returns:
    Path
        The path to the downloaded file.

    Raises:
    ValueError
        If the URL is invalid or malformed.
    URLError
        If there is an issue accessing the URL.
    OSError
        If there is an issue creating the output directory or saving the file.

    Note:
    This function assumes the presence of helper functions `_sanitize_url` and
    `safe_filename_from_url` for processing the URL and generating a safe file name.
    """
    sanitized = _sanitize_url(url)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = safe_filename_from_url(sanitized)
    destination = output_path / filename

    request = Request(
        sanitized,
        headers={"User-Agent": "PythonFileDownloader/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        with destination.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)

    return destination
