import os
from db.mysql_manager import MySQLManager
from managers.email_manager import EmailManager
from managers.events_manager import EventsManager
from managers.firebase_manager.firebase_manager import FirebaseManager
from managers.openai_manager import OpenAIManager
from managers.reportlab_manager.reportlab_manager import ReportlabManager
from managers.sentry_manager import SentryManager
from managers.square_manager import SquareManager
from managers.watchdog_manager import WatchdogManager
from utils.env_vars import ENV_VARS
from utils.local_deps import  local_deps
from utils.run_cron_tasks import CronTasksRunner
local_deps()


class DevConfigs:

  endpointMsgCodes = {
    'success':'OK',
    'error':'ERROR',
  }

  app ={}
  events_manager = EventsManager(ENV_VARS.get("EVENTBRITE_OAUTH_TOKEN"))
  # mysql_manager = MySQLManager(ENV_VARS.get("SQLALCHEMY_MYSQL_0_CONN_STRING"))
  email_manager= EmailManager(ENV_VARS.get("RESTDBIO_SERVER_API_KEY_0"))
  openai_manager = OpenAIManager(ENV_VARS.get("OPENAI_API_KEY_0"))
  sentry_manager = SentryManager()
  watchdog_manager = WatchdogManager()
  cron_task_runner =  CronTasksRunner()
  firebase_manager = FirebaseManager(ENV_VARS)
  reportlab_manager = ReportlabManager()
  square_manager = None

  def _create_app_info_obj(self,backend_port=5001):

    return {
      'access_control_allow_origin':['https://example.com:4200','https://example.com:4201'],
      'server_name':'example.com:{}'.format(backend_port),
      'domain_name':'https://example.com:{}'.format(backend_port),
      'flask_env':'development',
      'frontend_angular_app_url':'https://example.com:4201',
      'frontend_angular_app_domain':'example.com',
      'backend_port':backend_port
      # 'access_control_allow_origin':['https://localhost:4200','https://localhost:4201'],
      # 'server_name':'localhost:{}'.format(backend_port),
      # 'domain_name':'https://localhost:{}'.format(backend_port),
      # 'flask_env':'development',
      # 'frontend_angular_app_url':'https://localhost:4201',
      # 'frontend_angular_app_domain':'localhost',
      # 'backend_port':backend_port
    }

  def __init__(self):
    self.app =self._create_app_info_obj()
    self.square_manager = SquareManager(ENV_VARS.get("SQUARE_SANDBOX_ACCESS_TOKEN_0"),self)



class TestConfigs(DevConfigs):
  None

class PreviewConfigs(DevConfigs):

  def __init__(self) -> None:
    super().__init__()
    self.app['flask_env'] = 'production'
    self.app['access_control_allow_origin'] = ["https://ui.preview.yourapp.com"]
    self.app.pop('server_name')
    self.app.pop('domain_name')
    self.app['frontend_angular_app_url'] = "https://ui.preview.yourapp.com"
    self.app['frontend_angular_app_domain'] = "ui.preview.yourapp.com"

class ProdConfigs(DevConfigs):

  def __init__(self) -> None:
    super().__init__()
    self.app['flask_env'] = 'production'
    self.app['access_control_allow_origin'] = ["https://yourapp.com"]
    self.app.pop('server_name')
    self.app.pop('domain_name')
    self.app['frontend_angular_app_url'] = "https://yourapp.com"
    self.app['frontend_angular_app_domain'] = "www.yourapp.com"



CONFIGS:DevConfigs= {
  'PROD':lambda x:ProdConfigs(),
  'PREVIEW':lambda x:PreviewConfigs(),
  'DEV':lambda x:DevConfigs(),
  'TEST':lambda x:TestConfigs(),
}[ENV_VARS.get("FLASK_BACKEND_ENV")](None)











