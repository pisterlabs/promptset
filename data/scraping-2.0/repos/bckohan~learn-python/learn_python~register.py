import typer
import yaml
from enum import Enum
from typing import Optional, Union, Annotated
from learn_python.utils import ROOT_DIR, Singleton, git_push_file
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes
)
from cryptography.exceptions import InvalidSignature
import subprocess
from functools import cached_property
from requests import HTTPError
from termcolor import colored
from learn_python import main
from learn_python.utils import lp_logger, get_log_date
import time
import re
import warnings
from datetime import date
from learn_python.utils import LOG_DIR
import os


PRIVATE_KEY_FILE = ROOT_DIR / '.private_key.pem'
PUBLIC_KEY_FILE = ROOT_DIR / 'public_keys.pem'


class LLMBackends(Enum):
    """
    The LLM backends currently supported to run Delphi.
    """
    TEST    = 'test'
    OPEN_AI = 'openai'

    def __str__(self):
        return self.value

    @property
    def backend_class(self):
        if self.value == 'openai':
            from learn_python.delphi.openai import OpenAITutor
            return OpenAITutor
        elif self.value == 'test':
            from learn_python.delphi.test import TestAITutor
            return TestAITutor
        raise ValueError(f'Unrecognized backend: {self.value}')


class Config(Singleton):

    CONFIG_FILE = ROOT_DIR / '.config.yaml'

    server: Optional[str] = None
    enrollment: Optional[str] = None
    registered: bool = False
    _tutor: LLMBackends = LLMBackends.OPEN_AI

    private_key: Optional[PrivateKeyTypes] = None
    public_keys: Optional[PublicKeyTypes] = None

    @property
    def tutor(self):
        return self._tutor
    
    @tutor.setter
    def tutor(self, value):
        try:
            if value is not None:
                self._tutor = LLMBackends(value) if not isinstance(value, LLMBackends) else value
        except ValueError as err:
            warnings.warn(
                f'Unrecognized tutor driver: {value}. Defaulting to {self.tutor.value}'
            )
            
    @cached_property
    def student(self):
        """The student's name is fetched from their git config install"""
        try:
            result = subprocess.run(
                ['git', 'config', 'user.name'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                name = result.stdout.split()[0]
                if len(name) > 1:
                    return name
            # if no name, use email instead
            if self.student_email:
                return self.student_email.split('@')[0]
        except Exception:
            pass
        return 'Student'
    
    @cached_property
    def student_email(self):
        """The student's email is fetched from their git config install"""
        try:
            result = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True)
            return result.stdout
        except Exception:
            pass
        return None

    def __init__(self):
        if self.CONFIG_FILE.is_file():
            with open(self.CONFIG_FILE, 'r') as cfg:
                conf = yaml.safe_load(cfg)
                self.server = conf.get('server', self.server)
                self.registered = conf.get('registered', self.registered)
                self.enrollment = conf.get('enrollment', self.enrollment)
                self.tutor = conf.get('tutor', self.tutor)
        self.load_private_key()

    @cached_property
    def origin(self):
        """The forked github repository - this will be used as the unique ID for the student"""
        try:
            return subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True,
                text=True,
                cwd=ROOT_DIR
            ).stdout.strip()
        except Exception:
            lp_logger.exception('Unable to determine origin of repository.')
        return None

    def commit_count(self):
        try:
            return int(
                subprocess.check_output(
                    ['git', 'rev-list', '--all', '--count'],
                    cwd=ROOT_DIR
                ).strip()
            )
        except Exception:
            lp_logger.exception('Unable to determine commit count.')
        return None
    
    def commit_hash(self):
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=ROOT_DIR
            ).strip().decode('utf-8')
        except Exception:
            lp_logger.exception('Unable to determine commit hash.')
        return None

    def cloned_branch(self):
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=ROOT_DIR
            ).strip().decode('utf-8')
        except Exception:
            lp_logger.exception('Unable to determine cloned branch.')
        return None
    
    def is_registered(self):
        return PRIVATE_KEY_FILE.exists() and self.registered

    def load_private_key(self):
        if PRIVATE_KEY_FILE.is_file():
            with open(PRIVATE_KEY_FILE, 'rb') as f:
                contents = f.read()
                if contents:
                    self.private_key = serialization.load_pem_private_key(
                        contents,
                        password=None,
                        backend=default_backend()
                    )
        return self.private_key

    def load_public_keys(self):
        self.public_keys = []
        if PUBLIC_KEY_FILE.is_file():
            with open(PUBLIC_KEY_FILE, 'rb') as f:
                pem_data = f.read().decode()
            
            # Splitting the keys based on PEM headers/footers
            pem_keys = [
                f"-----BEGIN {m[1]}-----{m[2]}-----END {m[1]}-----"
                for m in re.findall(
                    r"(-----BEGIN (.*?)-----)(.*?)(-----END \2-----)",
                    pem_data,
                    re.S
                )
            ]

            self.public_keys = [
                serialization.load_pem_public_key(
                    pem.encode(),
                    backend=default_backend()
                ) for pem in pem_keys
            ]
        return self.public_keys

    def to_dict(self):
        return {
            'server': self.server,
            'registered': self.registered,
            'enrollment': self.enrollment,
            'tutor': self.tutor.value
        }
    
    def update(self, config: dict):
        self.server = config.get('server', self.server)
        self.registered = config.get('registered', self.registered)
        self.enrollment = config.get('enrollment', self.enrollment)
        self.tutor = config.get('tutor', self.tutor)
        return self

    def try_authorize_tutor(self):
        from learn_python.client import CourseClient
        client = CourseClient()
        try:
            tutor_auth = client.get_tutor_auth() or {}
        except HTTPError as err:
            if err.response.status_code == 403:
                tutor_auth = {}
            else:
                raise
        if 'tutor' in tutor_auth:
            self.tutor = tutor_auth['tutor']
            self.write()
            if 'secret' in tutor_auth:
                self.tutor.backend_class.write_key(tutor_auth['secret'])
                typer.echo(colored('Delphi has been authorized!', 'green'))
                return True
        return False
    
    def write(self):
        with open(self.CONFIG_FILE, 'w') as cfg:
            yaml.dump(self.to_dict(), cfg)
        return True
    
    def keys_valid(self):
        self.load_private_key()
        self.load_public_keys()
        if not (self.public_keys and self.private_key):
            return False
        
        def verify(key):
            msg = str(int(time.time())).encode()
            try:
                key.verify(
                    self.sign_message(msg),
                    msg,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except InvalidSignature:
                return False
        return any((verify(key) for key in self.public_keys))

    def register(self, reset: bool = False, timeout: int = 120):
        from learn_python.client import CourseClient
        client = CourseClient(timeout=timeout)

        if reset or not self.keys_valid():
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = self.private_key.public_key()

            with open(PUBLIC_KEY_FILE, 'w' if reset else '+a', encoding='utf-8') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8'))
            
            with open(PRIVATE_KEY_FILE, 'wb') as f:
                f.write(
                    self.private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                )

            git_push_file(PUBLIC_KEY_FILE)

        if not self.keys_valid():
            raise RuntimeError(
                'Unable to generate functioning key/value pair.'
            )
        
        try:
            client.register()
            lp_logger.info('Registered with course.')
        except HTTPError as err:
            lp_logger.exception('Unable to register with course.')
            return False
        
        try:
            self.try_authorize_tutor()
        except HTTPError as err:
            lp_logger.info('Tutor: %s authorized.', self.tutor)
            lp_logger.exception('Unable to authorize tutor.')
        return True

    def sign_message(self, message: Union[str, bytes]):
        signature = None
        if self.private_key:
            signature = self.private_key.sign(
                message.encode() if isinstance(message, str) else message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        return signature

@main(catch=False)
def register(
    reset: Annotated[
        bool,
        typer.Option(
            '--reset',
            help=(
                'Overwrite the existing public key file. If you have any other clones of this '
                'repository, this will unregister them from the course.'
            )
        )
    ] = False,
    timeout: Annotated[
        int,
        typer.Option(
            '--timeout',
            help='Timeout for the register request in seconds.'
        )
    ] = 120
):
    """
    Register for a guided course!
    """
    if Config().register(reset=reset, timeout=timeout):
        typer.echo(colored('Your course is now registered!', 'green'))
        if Config().enrollment is not None:
            typer.echo(colored(f'You have been enrolled in course: {Config().enrollment}', 'green'))
    else:
        typer.echo(
            colored('Course registration failed. If this is in error, contact your instructor.', 'red')
        )
    do_report()


# guard against excessive reporting
_report_lock = False


def lock_reporting(lock=True):
    global _report_lock
    _report_lock = lock


def can_report():
    global _report_lock
    return not _report_lock


@main(catch=True)
def report(
    keep: Annotated[
        bool,
        typer.Option(
            '-k',
            '--keep',
            help='Do not delete finished log files.'
        )
    ] = False,
    no_active: Annotated[
        bool,
        typer.Option(
            '--no-active',
            help='Do not report active log file.'
        )
    ] = False,
):
    """
    Report all status and logs to the course server. Logs for dates before now will be
    deleted.
    """
    do_report(keep=keep, no_active=no_active)


def do_report(keep=False, no_active=False):
    if not can_report():
        return
    try:
        from learn_python.client import CourseClient
        from learn_python.delphi.tutor import Tutor
        Tutor.submit_logs()
        course = CourseClient()
        date.today()
        for log_file in os.listdir(LOG_DIR):
            log_file = LOG_DIR / log_file
            dt = get_log_date(log_file)
            is_active = dt is None or dt == date.today()
            if is_active and no_active:
                continue
            
            try:
                course.post_log(log_file)
                if not keep and not is_active:
                    os.remove(log_file)
            except HTTPError as err:
                lp_logger.exception(
                    'Unable to post log file: %s',
                    log_file
                )
    except Exception:
        lp_logger.exception('Unable to report logs to course server.')


# these are needed for the docs, figure out if @main decorator is actually necessary
(register_app := typer.Typer(add_completion=False)).command()(register.__closure__[-1].cell_contents)
(report_app := typer.Typer(add_completion=False)).command()(report.__closure__[-1].cell_contents)
