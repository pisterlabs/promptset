#!/usr/bin/env python
#
# Simple test suite for bash2python_diff.py. As with test_bash2python.py, coming up with
# general tests will be difficult, given the open-ended nature of the task.
#

"""Tests for bash2python_diff.py"""

# Standard modules
# TODO

# Installed modules
import pytest
from click.testing import CliRunner

# Local modules
from bash2python import OPENAI_API_KEY
from bash2python_diff import main
from mezcla import debug
from mezcla.my_regex import my_re


@pytest.mark.skipif(not OPENAI_API_KEY, reason='OPENAI_API_KEY not set')
def test_diff_no_opts():
    """Tests bash2python_diff without flags enabled"""
    debug.trace(5, "test_diff_no_opts()")
    runner = CliRunner(mix_stderr=False)
    # NOTE: disables stderr tracing
    bash = "time\n\nuptime\n"
    result = runner.invoke(main, env={"DEBUG_LEVEL": "0"},
                           input=bash)
    python = result.output
    debug.trace_expr(5, bash, python, delim="\n")
    debug.trace_object(6, result)
    assert result.exit_code == 0
    assert "------------codex------------" in result.output
    assert "------------b2py------------" in result.output


@pytest.mark.skipif(not OPENAI_API_KEY, reason='OPENAI_API_KEY not set')
def test_diff_opts():
    """Tests bash2python_diff with all flags enabled"""
    debug.trace(5, "test_diff_opts()")
    runner = CliRunner(mix_stderr=False)
    # NOTE: disables stderr tracing
    bash = "echo Hello World;\nfoo='bar';\n"
    result = runner.invoke(main, args=["--perl", "--diff"], env={"DEBUG_LEVEL": "0"},
                           ## TODO2: input="echo Hello World\n\nfoo='bar'\n",
                           input=bash)
    python = result.output
    debug.trace_expr(5, bash, python, delim="\n")
    debug.trace_expr(6, result.output, max_len=4096)
    debug.trace_object(7, result)
    assert result.exit_code == 0
    # example output (simplified):
    #   # b2py                           | codex
    #   run('echo "Hello World"')        |
    #                                    | print("Hello World")
    #   fuu='bar'                        |
    #                                    | foo = 'bar'
    assert(my_re.search(r"""# b2py\s+\|\s+codex""", result.output))
    assert(my_re.search(r"""run\(.*echo Hello World.*\).*\|""", result.output))
    # TODO2: assert(my_re.search(r"""run\('echo Hello World'\).*\|""", result.output))
    assert(my_re.search(r"""\|.*print\("Hello World"\)""", result.output))
    assert(my_re.search(r"""foo\s*=\s*['"]bar['"].*\|""", result.output))
    assert(my_re.search(r"""\|.*foo\s*=\s*['"]bar['"]""", result.output))

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    debug.trace_current_context()
    pytest.main([__file__])
