r"""
 ____  _  _  ____  __  _  _   ___  ____  __  ___   _  _
(  __)( \/ )(_  _)(  )( \( ) / __)(_  _)(  )/   \ ( \( )
|  _)  )  (   )(   )( | ,` |( (__   )(   )((  O  )| ,` |
(____)(_/\_) (__) (__)(_)\_) \___) (__) (__)\___/ (_)\_)
  ________  ____  ____  ________  ___  ____  _________
 (_   __  \(_  _)(_  _)(_   __  \(   \(_   )(___   ___)
   | |_ \_|  \ \  / /    | |_ \_| |   \ | |     | |
   |  _| _    \ \/ /     |  _| _  | |\ \| |     | |
  _| |__/ |    \  /     _| |__/ | | |_\   |    _| |_
 (________/    (__)    (________/(____)\___)  (_____)

CEX - Centralized Exchange API Wrapper
Binance, Bitfinex, Bittrex, Coinbase, Kraken, Kucoin, Poloniex
Shared Utilities
litepresence 2019
"""

#
# DISABLE SELECT PYLINT TESTS
# pylint: disable=too-many-branches, too-many-statements, broad-except
#
# STANDARD MODULES
import functools
import json
import os
import time
import traceback
from calendar import timegm
from datetime import datetime
from json import dumps as json_dumps
from json import loads as json_loads
from math import ceil, floor, log10
from threading import Thread
import re

import numpy as np
import tulipy as tu

PATH = str(os.path.dirname(os.path.abspath(__file__))) + "/"
NIL = 10e-10

# FORMATTING
# ======================================================================
def red_to_green_fade(value):
    try:
        # Calculate the red and green components
        red = 255 - value  # Red decreases from 255 to 0
        green = value      # Green increases from 0 to 255

        # Construct the xterm256 color code
        color_code = 16 + (red // 51) * 36 + (green // 51) * 6
        return f"\033[38;5;{int(color_code)}m"
    except:
        return ""

def strip_ansi(string):
    return re.sub(r"\033\[.*?m", "", string)

def ljust_ansi(string, length):
    spaces = length - len(strip_ansi(string))
    if spaces > 0:
        string += " "*spaces
    return string

def print_table(data, x_pos=-1, y_pos=0, render=False, colors=None, pallete=None):
    """
    Prints a formatted table based on the provided data.

    Args:
        data (List[List[Union[float, str]]]):
            A rectangular table data represented as a list of rows,
            where each row is a list of values.
        pos (int, optional):
            The console position where the table should be printed.
            Default is -1, which means the table will be printed
            starting from the current console position.
    """

    # if sum(map(len, data)) != len(data[0]) * len(data):
    #     raise ValueError(f"Table must be rectangular.  Got lengths {list(map(len, data))}")
    # rotate table
    data2 = [[None for _ in data] for _ in data[0]]
    justs = []
    for cdx, column in enumerate(data):
        for celldx, cell in enumerate(column):
            if isinstance(cell, np.ndarray):
                items = functools.reduce(lambda x,y:x*y, cell.shape)
                if len(cell.shape) > 1 or items > 20:
                    column[
                        celldx
                    ] = f"<{len(cell.shape)}D array of {items} items>"
                else:
                    processed_cell = cell - np.min(cell)
                    processed_cell /= np.max(processed_cell)
                    column[
                        celldx
                    ] = "".join(f"{red_to_green_fade(value)}█\033[m" for value in np.clip(processed_cell*255, 0, 255))
        justs.append(
            max(
                len(str(sigfig(cell, 4) if isinstance(cell, float) else strip_ansi(str(cell))))
                for cell in column
            )
            + 2
        )
        for rdx, cell in enumerate(column):
            data2[rdx][cdx] = sigfig(cell, 4) if isinstance(cell, float) else cell
    # print table
    text = "" if x_pos < 0 else f"\033[{y_pos};{x_pos+1}H"
    for idx, row in enumerate(data2):
        sub = 0
        for cdx, cell in enumerate(row):
            if (x := len(strip_ansi(str(cell))) - justs[cdx]) > 0:
                sub = x
            if (x := justs[cdx] - sub) < 0:
                sub = x
            if colors is not None and int(colors[idx][cdx]):
                text += f"\033[38;5;{pallete[int(colors[idx][cdx])]}m{ljust_ansi(str(cell), justs[cdx] - sub)}\033[m"
            else:
                text += ljust_ansi(str(cell), justs[cdx] - sub)
        text += "\n" if x_pos < 0 else f"\033[{1+idx+y_pos};{x_pos+1}H"
    if not render:
        print(text)
    return text


def it(style, text):
    """
    Colored text in terminal using ANSI color codes
    """
    emphasis = {
        "black": 90,
        "red": 91,
        "green": 92,
        "yellow": 93,
        "blue": 94,
        "purple": 95,
        "cyan": 96,
        "white": 97,
        "default": 39,  # Default text color
    }

    # Check if the style is a valid key in the emphasis dictionary
    if style not in emphasis:
        raise ValueError(
            f"Invalid style: {style}. Available styles: {', '.join(emphasis.keys())}"
        )

    return f"\033[{emphasis[style]}m{str(text)}\033[m"


def block_print():
    """
    temporarily disable printing
    """
    sys.stdout = open(os.devnull, "w")


def enable_print():
    """
    re-enable printing
    """
    sys.stdout = sys.__stdout__


def print_elapsed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print(f"elapsed: {time.time() - start:.3f}")
        return ret

    return wrapper


def trace(error):
    """
    Stack trace report upon exception
    """
    # print(error.args)
    err = (
        "========================================================"
        + "\n\n"
        + str(time.ctime())
        + " "
        + str(type(error).__name__)
        + "\n\n"
        + str(error.args)
        + "\n\n"
        + str(traceback.format_exc())
        + "\n\n"
        + "========================================================"
    )
    print(err)
    return err


def sigfig(number, sig):
    try:
        return round(number, sig - int(floor(log10(abs(number)))) - 1) if number else 0
    except (ValueError, OverflowError):
        return number


def print_without_wif(order):
    print(
        json_dumps(
            {
                "edicts": order["edicts"],
                "header": {k: v for k, v in order["header"].items() if k != "wif"},
                "nodes": order["nodes"],
            },
            indent=4,
        )
    )


def format_timeframe(seconds):
    """
    take a value in seconds and return the next biggest candle size, i.e.
    1m, 5m, 1h, 8h, 1d, 1w, 1M, etc.
    """
    # minute, hour, day, week, month
    times = [60, 60 * 60, 60 * 60 * 24, 60 * 60 * 24 * 7, 60 * 60 * 24 * 7 * 4]
    # same, but in letter abbreviations
    labels = "mhdwM"
    for idx, (t, pt) in enumerate(zip(times[1:], times)):
        if t > seconds:
            return f"{ceil(seconds/pt)}{labels[idx]}"
    # return in months if overflow
    return f"{ceil(seconds/times[-1])}{labels[-1]}"


def unformat_timeframe(timeframe):
    """
    take a value like:
    1m, 5m, 1h, 8h, 1d, 1w, 1M, etc.
    and return it's value in seconds
    """
    # minute, hour, day, week, month
    times = {
        "m": 60,
        "h": 60 * 60,
        "d": 60 * 60 * 24,
        "w": 60 * 60 * 24 * 7,
        "M": 60 * 60 * 24 * 7 * 4,
    }
    return times[timeframe[-1]] * int(timeframe[:-1])


# FILES AND LOGGING
# ======================================================================
def read_file(path):
    """
    Read the contents of a file.

    Parameters:
    - path: The path to the file to read.

    Returns:
    - The contents of the file as a string.
    """
    with open(path, "r") as handle:
        data = handle.read()
    return data


def write_file(path, contents):
    """
    Write contents to a file in JSON format.

    Parameters:
    - path: The path to the file to write.
    - contents: The data to write to the file.
    """
    with open(path, "w") as handle:
        handle.write(json.dumps(contents, indent=1, cls=NdarrayEncoder))


def race_write(doc="", text=""):
    """
    Concurrent Write to File Operation
    """
    text = str(text)
    i = 0
    doc = os.path.join(PATH, "pipe", doc)
    while True:
        try:
            time.sleep(0.05 * i**2)
            i += 1
            with open(doc, "w+") as handle:
                handle.write(text)
                handle.close()
                break
        except Exception as error:
            msg = str(type(error).__name__) + str(error.args)
            msg += " race_write()"
            print(msg)
            try:
                handle.close()
            except:
                pass
            continue
        finally:
            try:
                handle.close()
            except:
                pass


def race_read(doc):
    """
    Concurrent Read JSON from File Operation
    """
    doc = os.path.join(PATH, "pipe", doc)
    i = 0
    while True:
        try:
            time.sleep(0.05 * i**2)
            i += 1
            with open(doc, "r") as handle:
                data = json_loads(handle.read())
                handle.close()
                return data
        except Exception as error:
            msg = str(type(error).__name__) + str(error.args)
            msg += " race_read_json()"
            print(msg)
            try:
                handle.close()
            except:
                pass
            if i > 5:
                raise error
            continue
        finally:
            try:
                handle.close()
            except:
                pass


def json_ipc(doc="", text="", initialize=False, append=False):
    """
    JSON IPC
    Concurrent Interprocess Communication via Read and Write JSON
    features to mitigate race condition:
        open file using with statement
        explicit close() in with statement
        finally close()
        json formatting required
        postscript clipping prevents misread due to overwrite without erase
        read and write to the text pipe with a single definition
        growing delay between attempts prevents cpu leak
    to view your live streaming database, navigate to the pipe folder in the terminal:
        tail -F your_json_ipc_database.txt
    :dependencies: os, traceback, json.loads, json.dumps
    :warn: incessant read/write concurrency may damage older spinning platter drives
    :warn: keeping a 3rd party file browser pointed to the pipe folder may consume RAM
    :param str(doc): name of file to read or write
    :param str(text): json dumped list or dict to write; if empty string: then read
    :return: python list or dictionary if reading, else None
    wtfpl2020 litepresence.com
    """
    # initialize variables
    data = None
    # file operation type for exception message
    if text:
        if append:
            act = "appending"
        else:
            act = "writing"
    else:
        act = "reading"
    # create a clipping tag for read and write operations
    tag = ""
    if not act == "appending":
        tag = "<<< JSON IPC >>>"
    # determine where we are in the file system; change directory to pipe folder
    path = os.path.join(PATH, "pipe")
    # ensure we're writing json then add prescript and postscript for clipping
    try:
        text = tag + json_dumps(json_loads(text)) + tag if text else text
    except Exception as error:
        print(text)
        print(error)
        raise ValueError
    # move append operations to the comptroller folder and add new line
    if append:
        path += "/comptroller"
        text = f"\n{text}"
    # create the pipe subfolder
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/comptroller", exist_ok=True)
    if doc:
        doc = f"{path}/{doc}"
        # race read/write until satisfied
        iteration = 0
        while True:
            # increment the delay between attempts exponentially
            time.sleep(0.02 * iteration**2)
            try:
                if act == "appending":
                    with open(doc, "a") as handle:
                        handle.write(text)
                        handle.close()
                        break
                elif act == "writing":
                    with open(doc, "w+") as handle:
                        handle.write(text)
                        handle.close()
                        break
                elif act == "reading":
                    with open(doc, "r") as handle:
                        # only accept legitimate json
                        data = json_loads(handle.read().split(tag)[1])
                        handle.close()
                        break
            except Exception:
                if iteration == 0 and act == "reading" and not os.path.exists(doc):
                    with open(doc, "w") as handle:
                        handle.write(f"{tag}{{}}{tag}")
                elif iteration == 5:
                    # maybe there is no pipe? auto initialize the pipe!
                    json_ipc(initialize=True)
                    print("json_ipc pipe initialized, retrying...\n")
                elif iteration == 10:
                    print("json_ipc unexplained failure\n", traceback.format_exc())
                iteration += 1
                continue
            # deliberately double check that the file is closed
            finally:
                try:
                    handle.close()
                except Exception:
                    pass
    return data


class NonceSafe:
    """
    ╔═══════════════════════════════╗
    ║ ╔╗╔╔═╗╔╗╔╔═╗╔═╗  ╔═╗╔═╗╔═╗╔═╗ ║
    ║ ║║║║ ║║║║║  ║╣   ╚═╗╠═╣╠╣ ║╣  ║
    ║ ╝╚╝╚═╝╝╚╝╚═╝╚═╝  ╚═╝╩ ╩╚  ╚═╝ ║
    ╚═══════════════════════════════╝

    context manager for process-safe nonce generation and inter process communication
        nonce generation
        process safe read
        process safe write
        process safe atomic read/write
    wtfpl litepresence.com 2022
    """

    @staticmethod
    def __enter__(*_) -> None:
        """
        file lock: try until success to change name of nonce.vacant to nonce.occupied
        """
        if not os.path.exists(f"{PATH}nonce_safe/nonce.vacant") and not os.path.exists(
            f"{PATH}nonce_safe/nonce.occupied"
        ):
            NonceSafe.restart()
        while True:
            # fails when nonce.occupied
            try:
                os.rename("nonce_safe/nonce.vacant", "nonce_safe/nonce.occupied")
                break
            except Exception:
                time.sleep(0.01)

    @staticmethod
    def __exit__(*_) -> None:
        """
        file unlock : change name of nonce.occupied back to nonce.vacant
        """
        os.rename("nonce_safe/nonce.occupied", "nonce_safe/nonce.vacant")

    @staticmethod
    def restart() -> None:
        """
        new locker: on startup, delete directory and start fresh
        """
        os.system(
            f"rm -r {PATH}nonce_safe; "
            + f"mkdir {PATH}nonce_safe; "
            + f"touch {PATH}nonce_safe/nonce.vacant"
        )
        thread = Thread(target=NonceSafe.free)
        thread.start()

    @staticmethod
    def free() -> None:
        """
        the nonce locker should never be occupied for more than a few milliseconds
        plausible the locker could get stuck, e.g. a process terminates while occupied
        """
        while True:
            # check every three seconds if the nonce is vacant
            if os.path.exists(f"{PATH}nonce_safe/nonce.vacant"):
                time.sleep(3)
            else:
                # check repeatedly for 3 seconds for vacancy
                start = time.time()
                while True:
                    elapsed = time.time() - start
                    if os.path.exists(f"{PATH}nonce_safe/nonce.vacant"):
                        break
                    # liberate the nonce locker
                    if elapsed > 3:
                        os.rename(
                            "nonce_safe/nonce.occupied", "nonce_safe/nonce.vacant"
                        )


class NdarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"numpy_array": obj.tolist(), "dtype": obj.dtype.str}
        return json.JSONEncoder.default(self, obj)


class NdarrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "numpy_array" in obj and "dtype" in obj:
            # Reconstruct the numpy array from the JSON object
            array_data = obj["numpy_array"]
            dtype_str = obj["dtype"]
            return np.array(array_data, dtype=dtype_str)
        return obj


# DATE UTILS
# ======================================================================
def from_iso_date(date):
    """
    ISO to UNIX conversion
    """
    return int(timegm(time.strptime(str(date), "%Y-%m-%dT%H:%M:%S")))


def to_iso_date(unix):
    """
    iso8601 datetime given unix epoch
    """
    return datetime.utcfromtimestamp(int(unix)).isoformat()


def to_short_iso_date(unix):
    """
    short datetime given unix epoch
    """
    return datetime.utcfromtimestamp(int(unix)).strftime("%Y/%m/%d")


def from_short_iso_date(date):
    """
    used for mode['end'] point of backtest ISO to UNIX conversion
    """
    return int(timegm(time.strptime(str(date), "%Y-%m-%d")))


def clock():
    """
    current 24 hour clock, local time, formatted HH:MM:SS
    """
    return str(time.ctime())[11:19]


def parse_date(date_str):
    # Check if the input is a Unix timestamp (integer or float)
    if isinstance(date_str, (int, float)):
        return date_str
    # Try parsing the date in the YY-MM-DD format
    try:
        return time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple())
    except ValueError:
        raise ValueError(
            f"Date must be in YYYY-MM-DD format or a Unix timestamp; got {date_str}"
        )


# MISC
# ======================================================================
def rotate(data):
    """
    Converts a list of dictionaries into a dictionary of lists and vice versa.
    """
    if isinstance(data, (list, np.ndarray)):
        if not np.any(data):  # Handle empty list case
            return {}

        # Transpose the values of the dictionaries into lists and return as a dictionary
        return {key: np.array([d[key] for d in data]) for key in data[0].keys()}

    elif isinstance(data, dict):
        if not data:  # Handle empty dictionary case
            return []

        # Create a list of dictionaries from the dictionary of lists
        return [
            {key: data[key][i] for key in data}
            for i, _ in enumerate(next(iter(data.values())))
        ]

    else:
        raise TypeError(
            f"Input must be a list of dictionaries or a dictionary of lists, got {type(data)} of {type(data[0])}s"
        )


def expand_bools(bool_list, side="both"):
    if not np.any(bool_list):
        return bool_list
    # Create a new list with the same length, initialized to False (or 0)
    lagged_list = [False] * len(bool_list)
    for i in range(len(bool_list) - 1):
        lagged_list[i] = (
            True
            if any(
                bool_list[i + j]
                for j in (
                    [-1, 0, 1]
                    if side == "both"
                    else [0, 1]
                    if side == "right"
                    else [-1, 0]
                )
            )
            else False
        )
    lagged_list[0] = bool_list[0]
    lagged_list[-1] = bool_list[-1]

    return lagged_list


def satoshi(number):
    """
    float prices rounded to satoshi
    """
    return float(f"{float(number):.8f}")


def satoshi_str(number):
    """
    string prices rounded to satoshi
    """
    return f"{float(number):.8f}"


def truncate(*args):
    """
    truncates multiple lists to the length of the shortest
    removing "oldest" data when newest is to the right
    """
    minlen = min(map(len, args))
    return tuple(i[-minlen:] for i in args)


class Period:
    pass


class FloatPeriod(float, Period):
    def __new__(cls, value):
        return super(FloatPeriod, cls).__new__(cls, value)


class IntPeriod(int, Period):
    def __new__(cls, value):
        return super(IntPeriod, cls).__new__(cls, value)


def period(num):
    if num.is_integer():
        return IntPeriod(num)
    else:
        return FloatPeriod(num)
