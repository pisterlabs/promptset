# -*- coding: utf-8
"""The commands.py module is used to generate XML API command tags.
"""

from __future__ import absolute_import

import xml.etree.ElementTree as ET

from oaxmlapi.base import _Base
from oaxmlapi.utilities import (READ_METHODS, REPORT_TYPES, SUBMIT_TYPES,
    SWITCH_TYPES, PAGE_ATTRIBUTES, APP_ATTRIBUTES, )


class Time(_Base):
    """
    The Time command returns the current time on our servers.

    Arguments:
        none

    """
    def __init__(self):
        _Base.__init__(self)

    def __str__(self):
        return "<Time>"

    def time(self):
        """
        Returns an ElementTree object containing a time tag.

        """
        return ET.Element('Time')

    def _main(self):
        return self.time()


class Read(_Base):
    """
    Use the read command to retrieve data from OpenAir.

    Arguments:
        type (str): a valid XML type
        method (str): a valid read method
        attribs (dict): a dictionary containing read attributes
        filters (list): a list of filter types
        orderby (list): a dictionary containing order attributes (field, order)
        return_fields (list): a list of fields to return

    """
    def __init__(self, type, method, attribs, filters=None, orderby=None, fields=None):
        _Base.__init__(self)
        self.type = type
        self.method = method
        self.attribs = attribs
        self.filters = filters
        self.orderby = orderby
        self.fields = fields

    def __str__(self):
        return "<Read type={type} method={method}>".format(
            type=self.type, method=self.method)

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, m):
        if not m in READ_METHODS:
            raise Exception('method "{method}" must be one of {allowed}'.format(
                method=m, allowed=READ_METHODS))
        self._method = m

    def read(self):
        """
        Returns an ElementTree object containing a read tag,
        as well as all the appropriate attributes and filters.

        """
        elem = ET.Element('Read')
        attribs = {}
        attribs['type'] = self.type
        attribs['method'] = self.method

        # process all read attributes
        for key in self.attribs:
            attribs[key] = self.attribs[key]

        # create filter information
        if self.filters:
            filter_list = []
            field_list = []

            for item in self.filters:
                if item['filter']:
                    filter_list.append(item['filter'])

                elif item['datatype']:  # pragma: no cover -> need to figure this out
                    elem.append(item['datatype'])

                if item['fieldname']:
                    field_list.append(item['fieldname'])
                    elem.append(item['datatype'])

            if field_list:
                attribs['field'] = ','.join(field_list)

            if filter_list:
                attribs['filter'] = ','.join(filter_list)

        # apply return order, if provided
        if self.orderby:
            field = self.orderby.get('field', 'id')
            order = self.orderby.get('order', 'asc')
            attribs['order'] = ','.join([field, order])

        # add all attribs to the XML element
        elem.attrib = attribs

        # process return fields
        if self.fields:
            subelem = ET.SubElement(elem, '_Return')

            for f in self.fields:
                ET.SubElement(subelem, f)

        return elem

    def _main(self):
        return self.read()


    class Filter(object):
        """
        Creates a filter object for filtering read commands.

        Arguments:
            filter (str): the type of filter
            field (str): the field to be filtered
            datatype (obj): a valid XML element

        """
        def __init__(self, filter, fieldname, datatype):
            self.filter = filter
            self.fieldname = fieldname
            self.datatype = datatype.getDatatype()

        def __str__(self):
            return "<Filter type={type} field={field}>".format(
                type=self.filter, field=self.fieldname)

        def getFilter(self):
            """
            Returns a dictionary of filter criteria.

            """
            f = {}
            f['filter'] = self.filter
            f['fieldname'] = self.fieldname
            f['datatype'] = self.datatype
            return f


class Report(_Base):
    """
    Use the Report command to run a report and email a PDF copy
    of a Timesheet, Envelope, or Saved report.

    Arguments:
        type (str): a valid XML type. Only Timesheet, Envelope
                        or Reportf datatypes are allowed
        datatype (obj): a valid XML report element datatype

    """
    def __init__(self, type, datatype):
        _Base.__init__(self)
        self.type = type
        self.datatype = datatype.getDatatype()

    def __str__(self):
        return "<Report type={type}>".format(type=self.type)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, t):
        if not t in REPORT_TYPES:
            raise Exception('type "{type}" must be one of {allowed}'.format(
                type=t, allowed=REPORT_TYPES))
        self._type = t

    def getReport(self):
        """
        Returns an ElementTree object containing report tags.

        """
        elem = ET.Element('Report')
        attribs = {}
        attribs['type'] = self.type
        elem.attrib = attribs
        elem.append(self.datatype)
        return elem

    def _main(self):
        return self.getReport()


class Add(_Base):
    """
    Use the Add command to add records.

    Arguments:
        type (str): a valid XML type
        attrib (dict): a dictionary containing add attributes
        datatype (obj): a valid Datatype() object

    """
    def __init__(self, type, attribs, datatype):
        _Base.__init__(self)
        self.type = type
        self.attribs = attribs
        self.datatype = datatype.getDatatype()

    def __str__(self):
        return '<Add type={type}>'.format(type=self.type)

    # type
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, t):
        if t == 'User':
            raise Exception('datatype "{type}" not supported - use CreateUser'.format(
                type=t))
        if t == 'Company':
            raise Exception('datatype "{type}" not supported - use CreateAccount'.format(
                type=t))
        self._type = t

    def add(self):
        """
        Returns an ElementTree object containing add tags.

        """
        elem = ET.Element('Add')
        attribs = {}
        attribs['type'] = self.type

        # process all add attributes
        for key in self.attribs:
            attribs[key] = self.attribs[key]

        elem.attrib = attribs
        elem.append(self.datatype)
        return elem

    def _main(self):
        return self.add()


class Delete(_Base):
    """
    Use the Delete command to delete records.

    Arguments:
        type (str): a valid XML type
        datatype (obj): a valid Datatype() object

    """
    def __init__(self, type, datatype):
        _Base.__init__(self)
        self.type = type
        self.datatype = datatype.getDatatype()

    def __str__(self):
        return '<Delete type={type}>'.format(type=self.type)

    def delete(self):
        """
        Returns an ElementTree object containing delete tags.

        """
        elem = ET.Element('Delete')
        attribs = {}
        attribs['type'] = self.type
        elem.attrib = attribs
        elem.append(self.datatype)
        return elem

    def _main(self):
        return self.delete()


class Modify(_Base):
    """
    Use the Modify command to change records.

    Arguments:
        type (str): a valid XML type
        attrib (dict): a dictionary containing modify attributes
        datatype (obj): a valid Datatype() object

    """
    def __init__(self, type, attribs, datatype):
        _Base.__init__(self)
        self.type = type
        self.attribs = attribs
        self.datatype = datatype.getDatatype()

    def __str__(self):
        return '<Modify type={type}>'.format(type=self.type)

    def modify(self):
        """
        Returns an ElementTree object containing modify tags.

        """
        elem = ET.Element('Modify')
        attribs = {}
        attribs['type'] = self.type

        # process all add attributes
        for key in self.attribs:
            attribs[key] = self.attribs[key]

        elem.attrib = attribs
        elem.append(self.datatype)
        return elem

    def _main(self):
        return self.modify()


class ModifyOnCondition(_Base):
    """
    Use the ModifyOnCondition command to perform actions such as
    updating the external_id of a record type only if the update
    time on the OpenAir server is older.

    Arguments:
        type (str): a valid XML type
        datatype1 (obj): a valid Datatype() object
        datatype2 (obj): a valid Datatype() object of type "Date"

    """
    def __init__(self, type, datatype1, datatype2):
        _Base.__init__(self)
        self.type = type
        self.datatype1 = datatype1.getDatatype()
        self.datatype2 = datatype2.getDatatype()

    def __str__(self):
        return '<ModifyOnCondition type={type}>'.format(type=self.type)

    def modify(self):
        """
        Returns an ElementTree object containing modify on condition tags.

        """
        elem = ET.Element('ModifyOnCondition')
        attribs = {}
        attribs['condition'] = "if-not-updated"
        attribs['type'] = self.type
        elem.attrib = attribs
        elem.append(self.datatype1)
        elem.append(self.datatype2)
        return elem

    def _main(self):
        return self.modify()


class Submit(_Base):
    """
    Use the Submit command to submit records.

    Arguments:
        type (str): a valid XML type
        datatype (obj): a valid Datatype() object
        approval (obj): a valid approval Datatype() object

    """
    def __init__(self, type, datatype, approval):
        _Base.__init__(self)
        self.type = type
        self.datatype = datatype.getDatatype()
        self.approval = approval.getDatatype()

    def __str__(self):
        return '<Submit type={type}>'.format(type=self.type)

    # type
    @property
    def type(self):
        return self._datatype

    @type.setter
    def type(self, t):
        if not t in SUBMIT_TYPES:
            raise Exception('type "{type}" must be one of {allowed}'.format(
                type=t, allowed=SUBMIT_TYPES))
        self._datatype = t

    def submit(self):
        """
        Returns an ElementTree object containing submit tags.

        """
        elem = ET.Element('Submit')
        attribs = {}
        attribs['type'] = self.type
        elem.attrib = attribs
        elem.append(self.datatype)
        elem.append(self.approval)
        return elem

    def _main(self):
        return self.submit()


class CreateAccount(_Base):
    """
    Use the CreateAccount command to create a new OpenAir account.
    When a new account is created, the first user is also created
    as the account administrator.

    Arguments:
        company (obj): a valid company Datatype() object
        user (obj): a valid user Datatype() object

    """
    def __init__(self, company, user):
        _Base.__init__(self)
        self.company = company
        self.user = user

    def __str__(self):
        return '<CreateAccount nickname={nickname}>'.format(
            nickname=self.company.fields['nickname'])

    # company
    @property
    def company(self):
        return self._company

    @company.setter
    def company(self, c):
        if not 'nickname' in c.fields:
            raise Exception('"nickname" is a required Company field')
        self._company = c

    # user
    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, u):
        if (
            not 'nickname' in u.fields or
            not 'password' in u.fields or
            not 'email' in u.fields
        ):
            raise Exception('"nickname, password and email" are required User fields')
        self._user = u

    def create(self):
        """
        Returns an ElementTree object containing submit tags.

        """
        elem = ET.Element('CreateAccount')
        elem.append(self.company.getDatatype())
        elem.append(self.user.getDatatype())
        return elem

    def _main(self):
        return self.create()


class CreateUser(_Base):
    """
    Use the CreateUser command to create a new OpenAir user.

    Arguments:
        company (obj): a valid company Datatype() object
        user (obj): a valid user Datatype() object

    """
    def __init__(self, company, user):
        _Base.__init__(self)
        self.company = company
        self.user = user

    def __str__(self):
        return '<CreateUser nickname={nickname}>'.format(
            nickname=self.user.fields['nickname'])

    # company
    @property
    def company(self):
        return self._company

    @company.setter
    def company(self, c):
        if not 'nickname' in c.fields:
            raise Exception('"nickname" is a required Company field')
        self._company = c

    # user
    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, u):
        if (
                not 'nickname' in u.fields or
                not 'password' in u.fields or
                not 'email' in u.fields
        ):
            raise Exception('"nickname, password and email" are required User fields')
        self._user = u

    def create(self):
        """
        Returns an ElementTree object containing submit tags.

        """
        elem = ET.Element('CreateUser')
        elem.append(self.company.getDatatype())
        elem.append(self.user.getDatatype())
        return elem

    def _main(self):
        return self.create()


class Switch(_Base):
    """
    Use the Switch command to set Company and User records.

    Arguments:
        type (str): a valid XML type
        datatype (obj): a valid flag Datatype object

    """
    def __init__(self, type, datatype):
        _Base.__init__(self)
        self.type = type
        self.datatype = datatype

    def __str__(self):
        return '<Switch type={type}>'.format(
            type=self.type)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, t):
        if not t in SWITCH_TYPES:
            raise Exception('type "{type}" must be one of {allowed}'.format(
                type=t, allowed=SWITCH_TYPES))
        self._type = t

    def switch(self):
        """
        Returns an ElementTree object containing a switch tag.

        """
        elem = ET.Element(self.type)
        flags = ET.SubElement(elem, 'flags')
        flags.append(self.datatype.getDatatype())
        return elem

    def _main(self):
        return self.switch()


class MakeURL(_Base):
    """
    Use the MakeURL command to obtain a URL for a specific application page.

    Arguments:
        uid (str): a user id string
        page (str): a page name string
        app (str): an app string
        arg (obj): a Datatype object

    """
    def __init__(self, uid, page, app, arg):
        _Base.__init__(self)
        self.uid = uid
        self.page = page
        self.app = app
        self.arg = arg

    def __str__(self):
        return "<MakeURL page={page} app={app}>".format(
            page=self.page, app=self.app)

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, p):
        if not p in PAGE_ATTRIBUTES:
            raise Exception('page "{page}" must be one of {allowed}'.format(
                page=p, allowed=PAGE_ATTRIBUTES))
        self._page = p

    @property
    def app(self):
        return self._app

    @app.setter
    def app(self, a):
        if not a in APP_ATTRIBUTES:
            raise Exception('app "{app}" must be one of {allowed}'.format(
                app=a, allowed=APP_ATTRIBUTES))
        self._app = a

    def makeurl(self):
        """
        Returns an ElementTree object containing an XML MakeURL tag.

        """
        makeurl = ET.Element('MakeURL')

        uid = ET.Element('uid')
        uid.text = self.uid

        page = ET.Element('page')
        page.text = self.page

        app = ET.Element('app')
        app.text = self.app

        if self.arg:
            arg = ET.Element('arg')
            arg.append(self.arg.getDatatype())
            makeurl.append(arg)

        makeurl.append(uid)
        makeurl.append(page)
        makeurl.append(app)
        return makeurl

    def _main(self):
        return self.makeurl()
