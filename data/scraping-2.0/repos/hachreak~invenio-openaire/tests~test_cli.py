# -*- coding: utf-8 -*-
#
# This file is part of Invenio.
# Copyright (C) 2015, 2016 CERN.
#
# Invenio is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# Invenio is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Invenio; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111-1307, USA.
#
# In applying this license, CERN does not
# waive the privileges and immunities granted to it by virtue of its status
# as an Intergovernmental Organization or submit itself to any jurisdiction.

"""CLI tests."""

from __future__ import absolute_import, print_function

from os.path import dirname, join

from click.testing import CliRunner
from invenio_pidstore.models import PersistentIdentifier

from invenio_openaire.cli import openaire


def test_loadfunders(script_info, es):
    """Test CLI for loading grants."""
    assert PersistentIdentifier.query.count() == 0
    runner = CliRunner()
    result = runner.invoke(
        openaire,
        ['loadfunders', '--source',
         join(dirname(__file__), 'testdata/fundref_test.rdf')],
        obj=script_info)
    assert result.exit_code == 0
    assert PersistentIdentifier.query.count() == 6


def test_loadgrants(script_info, es, funders):
    """Test CLI for loading grants."""
    # Funders only
    assert PersistentIdentifier.query.count() == 6
    runner = CliRunner()
    result = runner.invoke(
        openaire,
        ['loadgrants', '--source',
         join(dirname(__file__), 'testdata/openaire_test.sqlite')],
        obj=script_info)
    print(result.output)
    assert result.exit_code == 0
    assert PersistentIdentifier.query.count() == 46
