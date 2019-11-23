#!/usr/bin/env python3
"""
misc_utils.py

Utilities for filehandling and other minor things.

Based on the work of A. Loquercio et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)

Licensed under the MIT License (see LICENSE for details)
"""
import os
import json
import shutil
import os.path as osp
from util.defaults import _DEFAULT_ARGS, _DEFAULT_SETTINGS

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def get_experiment_folders(folder_in):
    """
    Gets all experiment subfolders if available.
    """
    folders = []
    if osp.isdir(osp.join(folder_in, 'images')):
        folders.append(folder_in)
    else:
        try:
            tmp = next(os.walk(osp.join(folder_in, '.')))[1]
            for f in tmp:
                folders.append(osp.join(folder_in, f))
        except IOError:
            print("Input folder not found.")
            exit(0)

    return sorted(folders)


def del_and_recreate_folder(folder):
    """
    Deletes a folder and its contens and then recreates a folder with the same name.
    """
    try:
        if osp.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    except IOError:
        print("Unable to create Folder.")


def write_to_file(dictionary, fname, beautify=False):
    """
    Writes everything in a dictionary to a json file.
    """
    with open(fname, "w") as f:
        json.dump(dictionary, f) if not beautify else json.dump(dictionary, f, sort_keys=True, indent=4)
        print("Written file {}".format(fname))


def load_settings_or_use_args(flags):
    """
    Loads settints either from stored file or arguments, with arguments always having the priority.
    :param flags:   Flags
    :return:        Settings as dict
    """

    # try to locate settings file at every reasonable location
    settings_filename = None
    sfname = flags.settings_fname or "_"
    rootdi = flags.experiment_rootdir or "_"
    inptdi = flags.input_dir or "_"
    possible_paths = [sfname, osp.join(rootdi, sfname), osp.join(inptdi, sfname)]

    for path in possible_paths:
        if osp.isfile(path):
            settings_filename = path
            break

    # load settings (with defaults as backup) if settings file found
    if settings_filename is not None:
        s = load_settings(settings_filename, defaults=_DEFAULT_SETTINGS)
    else:
        s = _DEFAULT_SETTINGS

    # always take args over loaded settings (if they were given) but fallback to defaults if necessary
    for key in _DEFAULT_ARGS:
        s[key] = flags.get_flag_value(key, None) or s.get(key, _DEFAULT_ARGS[key])

    # fix settings filename if path was give as input
    s['settings_fname'] = os.path.basename(s['settings_fname'])

    return s


def load_settings(filename, defaults=None):
    """
    Loads a settings dictionary from a json file and makes sure that all keys from the default options exist.
    If a key doesn't exist, the default value will be used.
    :param filename:
    :param defaults:
    :return:
    """

    try:
        with open(filename, "r") as read_file:
            s = json.load(read_file)
    except IOError:
        assert (defaults is not None), "Loading default settings failed."
        s = defaults
        print("File: {} not found, using defaults instead.".format(filename))

    # check if all values exist in loaded settings, otherwise load from defaults
    if defaults is not None:
        for k in defaults:
            s[k] = s.get(k, defaults[k])

    return s
