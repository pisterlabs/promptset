from typing import List, Optional, Tuple
from subprocess import Popen, PIPE
from dotenv import load_dotenv
import os
from datetime import datetime
import time
import openai  # type: ignore

from utils import expect_env
from models import API, Endpoint
from generate import generate_app, run_necessary_migrations


load_dotenv()
openai.api_key = expect_env('OPENAI_API_KEY')

class Manager:
    api: Optional[API] = None
    proc: Optional[Popen[bytes]] = None

    def set_api(self, api: API):
        self.api = api

    def restart(self):
        self.stop()
        self.start()

    def start(self) -> None:
        self.regenerate_code()
        if self.proc is not None:
            raise ValueError('Manager Error: process is already running')
        self.proc = Popen(['uvicorn', 'generated.app:app', '--host', '0.0.0.0', '--port', '3002'], stdout=PIPE, stderr=PIPE)
        time.sleep(2)  # TODO: fix this race condition properly
        print('Manager: successfully started API')

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None
        print('Manager: successfully stopped API')

    def wait(self) -> Tuple[bytes, bytes]:
        if self.proc is None:
            raise ValueError('Manager Error: cannot wait on process because it has not been started')
        stdout, stderr = self.proc.communicate()
        return stdout, stderr

    def regenerate_code(self):
        if self.api is None:
            raise ValueError('Manager Error: api has not been set')
        if not os.path.exists('generated'):
            os.mkdir('generated')
        code = generate_app(self.api.title, self.api.endpoints)
        with open('generated/app.py', 'w') as f:
            f.write(code)

    def clear_api(self):
        self.api = None

    def apply_migrations(self, migrations: str):
        run_necessary_migrations(sql_migrations=)


if __name__ == '__main__':
    m = Manager()
    a = API(
        id=1,
        title='My API',
        endpoints=[],
        user_id=1,
        created=datetime.utcnow(),
        updated=datetime.utcnow(),
    )
    m.set_api(a)
    m.start()
    time.sleep(2)
    m.stop()
    out, err = m.wait()
    print(out, err)
