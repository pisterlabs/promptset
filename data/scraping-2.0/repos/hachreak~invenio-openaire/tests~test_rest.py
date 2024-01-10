# -*- coding: utf-8 -*-
#
# This file is part of Invenio.
# Copyright (C) 2016 CERN.
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

"""Tests for OpenAIRE dataset loaders and resolvers."""

from __future__ import absolute_import, print_function

from invenio_records_rest import InvenioRecordsREST

from invenio_openaire.config import OPENAIRE_REST_ENDPOINTS


def test_records_rest(app, db, es, grants):
    """Test Records REST."""
    app.config['RECORDS_REST_ENDPOINTS'] = OPENAIRE_REST_ENDPOINTS
    app.config['RECORDS_REST_DEFAULT_READ_PERMISSION_FACTORY'] = None
    InvenioRecordsREST(app)

    with app.test_client() as client:
        # Item
        res = client.get("/funders/10.13039/001")
        assert res.status_code == 200
        # List
        res = client.get("/funders/")
        assert res.status_code == 200
        print(res.get_data(as_text=True))
        # Suggest
        res = client.get("/funders/_suggest?text=Uni")
        assert res.status_code == 200

        # Item
        res = client.get("/grants/10.13039/501100000923::LP0667725")
        assert res.status_code == 200
        # List
        res = client.get("/grants/")
        assert res.status_code == 200
        # Suggest
        res = client.get(
            "/grants/_suggest?text=open&funder=10.13039/501100000923")
        assert res.status_code == 200
