#!/usr/bin/env python

"""Tests for the ``api_key`` property of the ``embed`` module."""

import contextlib
import logging
import os
from pathlib import Path
import string
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dulwich.porcelain
import openai
from parameterized import parameterized

import embed
from embed._keys import _get_key_if_available
from tests import _bases

if sys.version_info < (3, 11):
    @contextlib.contextmanager
    def _chdir(path):
        """Trivial non-reentrant version of ``contextlib.chdir`` for < 3.11."""
        _old_pwd = Path().absolute()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(_old_pwd)
else:
    _chdir = contextlib.chdir


class TestApiKey(_bases.TestBase):
    """Tests for ``embed.api_key``."""

    def setUp(self):
        """
        Save ``api_key`` attributes. Also pre-patch them, for log redaction.
        """
        super().setUp()

        # This cannot be done straightforwardly with unittest.mock.patch
        # because that expects to be able to delete attributes, and the
        # embed.api_key property (deliberately) has no deleter.
        self._real_key_openai = openai.api_key
        self._real_key_embed = embed.api_key
        openai.api_key = 'sk-fake_redact_outer'
        embed.api_key = 'sk-fake_redact_inner'

    def tearDown(self):
        """Unpatch ``api_key`` attributes."""
        embed.api_key = self._real_key_embed
        openai.api_Key = self._real_key_openai

        super().tearDown()

    @parameterized.expand([
        ('str', 'sk-fake_setting_sets'),
        ('none', None),
    ])
    def test_setting_on_embed_sets_on_openai(self, _name, pretend_key):
        """Setting ``embed.api_key`` sets both it and ``openai.api_key``."""
        embed.api_key = pretend_key
        with self.subTest('embed.api_key'):
            self.assertEqual(embed.api_key, pretend_key)
        with self.subTest('openai.api_key'):
            self.assertEqual(openai.api_key, pretend_key)

    @parameterized.expand([
        ('str', 'sk-fake_setting_does_not_set'),
        ('none', None),
    ])
    def test_setting_on_openai_does_not_set_on_embed(self, _name, pretend_key):
        """Setting ``open.api_key`` does not change ``embed.api_key``."""
        openai.api_key = pretend_key
        self.assertNotEqual(embed.api_key, pretend_key)


_ONE_LETTER_DIR_NAMES = tuple(string.ascii_lowercase)
"""Directory names for testing, for below the point of interest."""

_TWO_CAP_LETTER_DIR_NAMES = tuple(ch * 2 for ch in string.ascii_uppercase)
"""Directory names for testing, for above the point of greatest interest."""

_THREE_LETTER_DIR_NAMES = tuple(ch * 3 for ch in string.ascii_lowercase)
"""Directory names for testing, higher above the point of greatest interest."""


def _create_and_enter_single_directory(name):
    """``mkdir`` one new subdirectory and ``cd`` into it."""
    subdir = Path(name)
    subdir.mkdir()
    os.chdir(subdir)


def _create_and_enter_directories(names):
    """``mkdir`` each new subdirectory and ``cd`` into it. Builds a chain."""
    for name in names:
        _create_and_enter_single_directory(name)


def _write_fake_key_file(fake_key):
    """Write a ``.api_key`` file with a fake key, in the current directory."""
    Path('.api_key').write_text(fake_key, encoding='utf-8')


class TestGetKeyIfAvailable(_bases.TestBase):
    """
    Tests for the non-public ``embed._keys._get_key_if_available`` function.

    These tests test the code that is used to determine the automatic initial
    value of ``embed.api_key``.

    The implementation logs extensively, but this does not currently test that.
    """

    def setUp(self):
        """
        Use a temporary directory; patch ``OPENAI_API_KEY``; quiet some logs.
        """
        super().setUp()

        # Create a temporary directory (that will be cleaned up) and cd to it.
        # pylint: disable-next=consider-using-with
        self.enterContext(_chdir(self.enterContext(TemporaryDirectory())))

        # Patch OPENAI_API_KEY to a fake value in the environment.
        environ_fragment = {'OPENAI_API_KEY': 'sk-fake_from_env'}
        self.enterContext(patch.dict(os.environ, environ_fragment))

        # Temporarily suppress embed._keys log messages less severe than error.
        logger = logging.getLogger(embed._keys.__name__)
        self.enterContext(patch.object(logger, 'level', logging.ERROR))

    # pylint: disable=missing-function-docstring  # Tests' names describe them.

    def test_uses_env_var_when_no_key_file(self):
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_env')

    def test_uses_env_var_instead_of_key_file(self):
        _write_fake_key_file('sk-fake_from_file')
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_env')

    def test_uses_key_file_in_cwd_when_no_env_var(self):
        del os.environ['OPENAI_API_KEY']
        _write_fake_key_file('sk-fake_from_file')
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_file')

    def test_none_found_when_no_env_var_nor_key_file(self):
        del os.environ['OPENAI_API_KEY']
        result = _get_key_if_available()
        self.assertIsNone(result)

    def test_key_file_in_parent_when_no_repo_not_used(self):
        del os.environ['OPENAI_API_KEY']
        _write_fake_key_file('sk-fake_from_file')
        _create_and_enter_single_directory('subdir')
        result = _get_key_if_available()
        self.assertIsNone(result)

    @parameterized.expand([
        (f'{above}above_{below}below', above, below)
        for below in (0, 1, 2, 5) for above in (1, 2)
    ])
    def test_key_file_outside_repo_not_used(self, _name, above, below):
        del os.environ['OPENAI_API_KEY']
        _write_fake_key_file('sk-fake_from_file')
        _create_and_enter_directories(_TWO_CAP_LETTER_DIR_NAMES[:above])
        dulwich.porcelain.init()
        _create_and_enter_directories(_ONE_LETTER_DIR_NAMES[:below])
        result = _get_key_if_available()
        self.assertIsNone(result)

    @parameterized.expand([
        (f'{above}above_{below}below', above, below)
        for below in (0, 1, 3) for above in (0, 1, 3)
    ])
    def test_key_file_inside_repo_used_when_no_env_var(self, _name,
                                                       above, below):
        del os.environ['OPENAI_API_KEY']
        dulwich.porcelain.init()
        _create_and_enter_directories(_TWO_CAP_LETTER_DIR_NAMES[:above])
        _write_fake_key_file('sk-fake_from_file')
        _create_and_enter_directories(_ONE_LETTER_DIR_NAMES[:below])
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_file')

    @parameterized.expand([
        (f'{above}above_{between}between_{below}below', above, between, below)
        for below in (0, 1, 3) for between in (1, 2, 4) for above in (0, 1, 3)
    ])
    def test_key_file_in_outer_nested_repo_not_used(self, _name,
                                                    above, between, below):
        del os.environ['OPENAI_API_KEY']
        _write_fake_key_file('sk-fake_from_file')
        _create_and_enter_directories(_THREE_LETTER_DIR_NAMES[:above])
        dulwich.porcelain.init()  # Outer enclosing repo.
        _create_and_enter_directories(_TWO_CAP_LETTER_DIR_NAMES[:between])
        dulwich.porcelain.init()  # Inner enclosed ("current") repo.
        _create_and_enter_directories(_ONE_LETTER_DIR_NAMES[:below])
        result = _get_key_if_available()
        self.assertIsNone(result)

    @parameterized.expand([
        (f'{above}above_{between}between_{below}below', above, between, below)
        for below in (0, 1, 3) for between in (0, 1, 3) for above in (1, 2, 4)
    ])
    def test_key_file_in_inner_nested_repo_used_when_no_env_var(
        self, _name, above, between, below,
    ):
        del os.environ['OPENAI_API_KEY']
        dulwich.porcelain.init()  # Outer enclosing repo.
        _create_and_enter_directories(_THREE_LETTER_DIR_NAMES[:above])
        dulwich.porcelain.init()  # Inner enclosed ("current") repo.
        _create_and_enter_directories(_TWO_CAP_LETTER_DIR_NAMES[:between])
        _write_fake_key_file('sk-fake_from_file')
        _create_and_enter_directories(_ONE_LETTER_DIR_NAMES[:below])
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_file')

    def test_prefers_env_var_to_any_files_in_repo(self):
        _write_fake_key_file('sk-fake_from_file_parent')
        dulwich.porcelain.init()
        _create_and_enter_single_directory('subdir')
        _write_fake_key_file('sk-fake_from_file_current')
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_env')

    def test_prefers_current_dir_to_parent_in_repo(self):
        del os.environ['OPENAI_API_KEY']
        _write_fake_key_file('sk-fake_from_file_parent')
        dulwich.porcelain.init()
        _create_and_enter_single_directory('subdir')
        _write_fake_key_file('sk-fake_from_file_current')
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_file_current')

    @parameterized.expand([
        ('no_prefix', 'fake_from_file'),
        ('non_word_chars', 'sk-fake+from+file'),
    ])
    def test_malformed_key_not_used_from_key_file(self, _name, malformed_key):
        del os.environ['OPENAI_API_KEY']
        _write_fake_key_file(malformed_key)
        result = _get_key_if_available()
        self.assertIsNone(result)

    @parameterized.expand([
        ('no_prefix', 'fake_from_file_current'),
        ('non_word_chars', 'sk-fake+from+file+current'),
    ])
    def test_skips_malformed_key_file_falls_back_to_ancestor_in_repo(
        self, _name, malformed_key,
    ):
        del os.environ['OPENAI_API_KEY']
        _write_fake_key_file('sk-fake_from_file_parent')
        dulwich.porcelain.init()
        _create_and_enter_single_directory('subdir')
        _write_fake_key_file(malformed_key)
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_file_parent')


if __name__ == '__main__':
    unittest.main()
