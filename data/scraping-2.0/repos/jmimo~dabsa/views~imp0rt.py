from baseview import BaseView
from web import app
from flask import request
from flask.ext.login import login_required
from database import db
from openair import parse
from model import AirspaceFile
import dlog

class ImportView(BaseView):
    methods = ['GET', 'POST']

    def get_template_name(self):
        return 'import.html'
    
    @login_required
    def dispatch_request(self):

        logger = dlog.get_logger('imp0rt')

        if request.method == 'POST':
            importfile = request.files['airspace']
            if importfile:
                logger.info('parsing file')
                airspaceFile = parse(importfile.filename,importfile)
                logger.info('file parsed')
                try:
                    db.add(airspaceFile)
                    logger.info('added airspaces to database')
                    db.commit()
                    logger.info('commited data')
                except Exception as e:
                    logger.error(e.message)
                    logger.info(e)
                    raise

        model = self.get_objects()
        model['files'] = AirspaceFile.query.all()
        return self.render_template(model)


app.add_url_rule('/import', view_func=ImportView.as_view('importview'))
