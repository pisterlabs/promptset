'''
implementation of ws, mq and http protocols
'''

import os
from os.path import join, dirname, splitext, isabs
import time
import shutil
import re
import yaml
from typing import Callable, Awaitable, Dict, Any, List, Set, Union, Tuple
from logging import getLogger, Logger
import json
import asyncio
import aiofiles
import logging
import glob

import websockets
from websockets.legacy.server import WebSocketServerProtocol
import aio_pika
from aiohttp import web
import openai

from agi_framework.dispatcher import Protocol, format_call
from agi_framework.config import Config

from queue import Queue
from os.path import exists
import ast
import gc

here = dirname(__file__)
logger = logging.getLogger(__name__)
log_level = os.getenv('LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=log_level)

# RabbitMQ port 5672
# VScode debug port 5678
# Browser port -p option (default=8000)
# WebSocket port is browser port + 1 (default=8001)

text_content_types = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
}

class WebSocketProtocol(Protocol):
    '''
    Websocket server
    '''
    protocol_id: str = 'ws'

    def __init__(self, host:str='0.0.0.0', port:int=8001, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.socket = None
        if host=='localhost':
            self.origins = [f'http://localhost:{port-1}']
        else:
            self.origins = None

    async def arun(self):
        asyncio.create_task(super().arun())

        if self.is_server:
            self.connected = set()
            await websockets.serve(
                self.handle_connection, self.host, self.port,
                origins=self.origins,
                )

    async def aclose(self):
        await super().aclose()

        if self.is_server:
            # Close all WebSocket connections
            for ws in self.connected:
                await ws.close()
            self.connected.clear()

    async def handle_connection(self, websocket: WebSocketServerProtocol, path):
        'Register websocket connection and redirect messages to the node instance'
        # Set CORS headers
#        origin = websocket.request_headers.get('Origin')

#        if origin in ['http://localhost:8000']:  # Adjust this as per your requirements
#            logger.info(f'Allowing CORS for {origin}')
#            websocket.response_headers.append(('Access-Control-Allow-Origin', origin))

        self.connected.add(websocket)
        try:
            node:Protocol = await self.handle_mesg('connect')
            if not isinstance(node, Protocol):
                logger.error('No Protocol node returned from on_ws_connect')
                return
            node_ws = node.get_protocol('ws')
            node_ws.socket = websocket

            await node_ws.handle_mesg('connect')

            async for mesg in websocket:
                data = json.loads(mesg)
                await node_ws.handle_mesg(**data)

        finally:
            # Unregister websocket connection
            await node_ws.handle_mesg('disconnect')
            self.connected.remove(websocket)

    async def do_send(self, cmd:str, **kwargs):
        'send ws message to browser via websocket'
        kwargs['cmd'] = cmd
        if self.socket:
            await self.socket.send(json.dumps(kwargs))

class HTTPProtocol(Protocol):
    '''
    http server
    Use port for http server
    Use port+1 for websocket port
    '''
    protocol_id: str = 'http'

    def __init__(self, root:str, host:str='0.0.0.0', port:int=8000, nocache=False, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.host = host
        self.port = port
        self.app:web.Application = None
        self.runner:web.AppRunner = None
        self.site:web.TCPSite = None
        self.md_content = None
        self.substitutions = {}
        self.static = [join(here, 'static')]
        self.static_handlers:List[Callable] = []

        if nocache:
            # force browser to reload static content
            self.substitutions['__TIMESTAMP__'] = str(time.time())

    def add_static(self, path:str):
        'add static directory'
        if not exists(path):
            logger.warn(f'Static directory {path}: does not exist')
        self.static.append(path)

    def add_static_handler(self, handler:Callable):
        'add static handler'
        self.static_handlers.append(handler)

    def find_static(self, filename:str):
        for static_dir in self.static:
            file_path = os.path.join(static_dir, filename)
            if os.path.isfile(file_path):
                return file_path
        return None

    def find_static_glob(self, filename:str):
        files = []
        for static_dir in self.static:
            file_path = os.path.join(static_dir, filename)
            files.extend(glob.glob(file_path))
        return files

    def index_md(self):
        index_file = join(self.static[:2][-1], 'docs', 'index.md')
        files = self.find_static_glob('docs/*.md')
        newest_file = max(files, key=os.path.getmtime)

        if newest_file != index_file:
            if index_file in files:
                files.remove(index_file)
            files.sort()

            with open(index_file, 'w') as f:
                f.write('| File | Description |\n')
                f.write('| ---- | ----------- |\n')
                for file in files:
                    with open(file, 'r') as f2:
                        s = f2.read()
                        # find first markdown header
                        m = re.search(r'^#+\s+(.*)', s, re.MULTILINE)
                        if m:
                            header = m.group(1)
                            base = os.path.basename(file).replace('.md','')
                            f.write(f'| [**{base}**](/docs/{base}) | *{header}* |\n')


        if not exists(index_file):
            return None
        return index_file

    async def handle_static(self, request:web.Request):
        filename = request.match_info['filename'] or 'index.html'
        query = request.query.copy()

        if filename == 'docs':
            file_path_md = self.index_md()
            filename = 'docs/index'
        else:
            # check for filename+'.md' and serve that instead with query: view=render
            file_path_md = self.find_static(filename+'.md')

        if file_path_md is not None:
            query.add('view','render')
            file_path = file_path_md
            filename = filename+'.md'
        else:
            file_path = self.find_static(filename)

        if file_path is None:
            for h in self.static_handlers:
                file_path = h(filename)
                if file_path is not None:
                    break
            else:
                return web.HTTPNotFound()

        ext = os.path.splitext(filename)[1]
        content_type = text_content_types.get(ext, None) # None means binary

        if content_type == 'text/markdown':
            format = query.get('view', 'raw')

            if format == 'raw':
                return web.FileResponse(file_path)

            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                raise web.HTTPNotFound()

            with open(join(here,'md_template.html'), 'r') as template_file:
                template = template_file.read()

            for key, value in self.substitutions.items():
                template = template.replace(key, value)

            template = template.replace('__MD_FILE__', filename)
            template = template.replace('__MD_VIEW__', format)

            return web.Response(text=template, content_type='text/html')

        # If substitutions are required for textual files
        elif content_type is not None:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                text = await f.read()

            if self.substitutions:
                for key, value in self.substitutions.items():
                    text = text.replace(key, value)

                response = web.Response(text=text, content_type=content_type)
                return response
        return web.FileResponse(file_path)

    async def arun(self):
        # Serve static http content from index.html
        asyncio.create_task(super().arun())

        self.app = web.Application()
        self.app.router.add_get('/{filename:.*}', self.handle_static)
        self.app.router.add_get('/', self.handle_static, name='index')
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        logger.info(f'Serving http://{self.host}:{self.port}')
        await self.site.start()

    async def aclose(self):
        # Stop the aiohttp site
        if self.site:
            await self.site.stop()

        # Shutdown and cleanup the aiohttp app
        if self.app:
            await self.app.shutdown()
            await self.app.cleanup()

        # Finally, cleanup the AppRunner
        if self.runner:
            await self.runner.cleanup()

        await super().aclose()


    async def on_ws_request_md_content(self):
        'request markdown content from browser via websocket'
        http = self.get_protocol('http')

        if http.md_content is not None:
            await self.send('ws', 'update_md_content', content=http.md_content, format=http.md_format)


class RabbitMQProtocol(Protocol):
    '''
    RabbitMQ broadcast protocol
    '''

    protocol_id: str = 'mq'

    def __init__(self, host:str, port:int=5672, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.connection: aio_pika.Connection = None
        self.channel: aio_pika.Channel = None
        self.exchange: aio_pika.Exchange = None
        self.queues: Dict[str, aio_pika.Queue] = {}  # Store queues per channel
        self.offline_queue: Queue = Queue() # queue for messages pending connection
        self.offline_subscription_queue: Queue = Queue() # queue for subscriptions pending connection
        self.connected = False

    async def arun(self):
        asyncio.create_task(super().arun())

        try:
            logger.info(f'Connecting to RabbitMQ on {self.host}:{self.port}')
            self.connection = await aio_pika.connect_robust(host=self.host, port=self.port)
        except aio_pika.AMQPException as e:
            logger.error(f"RabbitMQ connection failed: {e}")
            return

        self.channel = await self.connection.channel()
        self.exchange = await self.channel.declare_exchange('agi.green', aio_pika.ExchangeType.DIRECT)
        self.connected = True

        logger.info(f'Connected to RabbitMQ on {self.host}:{self.port}')

        # Do any pending subscriptions
        while not self.offline_subscription_queue.empty():
            channel_id = self.offline_subscription_queue.get()
            await self.subscribe(channel_id)

        # Send any pending messages
        while not self.offline_queue.empty():
            cmd, ch, kwargs = self.offline_queue.get()
            await self.do_send(cmd, ch, **kwargs)


    async def aclose(self):
        # Close the RabbitMQ channel and connection
        await self.unsubscribe_all()

        if self.channel:
            await self.channel.close()
            await self.connection.close()

        # terminate

        await super().aclose()

    async def listen_to_queue(self, channel_id, queue):
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    data = json.loads(message.body.decode())
                    if data['cmd'] == 'unsubscribe':
                        if data['sender_id'] == id(self):
                            break
                    else:
                       await self.handle_mesg(channel_id=channel_id, **data)

        del self.queues[channel_id]
        logger.info(f'{self._root.username} unsubscribed from {channel_id}')

    async def subscribe(self, channel_id: str):
        if not self.connected:
            self.offline_subscription_queue.put(channel_id)
            return

        if channel_id not in self.queues:
            queue = await self.channel.declare_queue(exclusive=True)
            await queue.bind(self.exchange, routing_key=channel_id)
            self.queues[channel_id] = queue
            logger.info(f'{self._root.username} subscribed to {channel_id}')

            asyncio.create_task(self.listen_to_queue(channel_id, queue))

    async def unsubscribe(self, channel_id: str):
        await self.send('mq', 'unsubscribe', channel=channel_id, sender_id=id(self))

    async def unsubscribe_all(self):
        'unsubscribe to everything'
        for channel_id in list(self.queues.keys()):
            await self.unsubscribe(channel_id)


    async def do_send(self, cmd: str, channel: str, **kwargs):
        'broadcast message to RabbitMQ'
        if not self.connected:
            self.offline_queue.put((cmd, channel, kwargs))
            return

        kwargs['cmd'] = cmd

        await self.exchange.publish(
            aio_pika.Message(body=json.dumps(kwargs).encode()),
            routing_key=channel  # We use routing key as channel_id for direct exchanges
        )



openai_key_request = '''
OpenAI API key required for GPT-4 chat mode.

``` form
fields:
  - type: text
    label: OpenAI API key
    key: openai.key
```
'''


class GPTChatProtocol(Protocol):
    '''
    OpenAI GPT Chat protocol

    This is just a POC: simple async wrapper around the OpenAI API in chat mode.
    Next step is to implement HuggingFace transformers and langchain for more control.
    '''
    protocol_id: str = 'gpt'

    def __init__(self, config:Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.name = 'agi.green'
        self.uid = 'bot'

    async def arun(self):
        asyncio.create_task(super().arun())

        # Ensure the OpenAI client is authenticated
        api_key = os.environ.get("OPENAI_API_KEY", None)

        if api_key is None:
            raise Exception("OPENAI_API_KEY environment variable must be set")

        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        if api_key:
            openai.api_key = api_key
        else:
            logger.warn('Missing OpenAI API key in config')

    async def on_ws_form_data(self, cmd:str, data:dict):
        key = data.get('key')
        openai.api_key = key
        self.config.set('openai.key', key)
        self.messages.append({"role": "system", "content": "OpenAI API key was just now set by the user."})
        await self.get_completion()

    async def on_ws_connect(self):
        await self.send('ws', 'set_user_data', uid='bot', name='GPT-4', icon='images/bot.png')
        await self.send('ws', 'set_user_data', uid='info', name='InfoBot', icon='images/bot.png')

    async def on_mq_chat(self, author:str, content:str):
        'receive chat message from RabbitMQ'
        if author != self.uid:
            self.messages.append({"role": "user", "content": content})
            task = asyncio.create_task(self.get_completion())

    async def get_completion(self):
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, self.sync_completion)
        await self.send('mq', 'chat', channel='chat.public', author=self.uid, content=content)

    def sync_completion(self):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=self.messages,
                )
            return response.choices[0]['message']['content']
        except Exception as e:
            msg = f'OpenAI API error: {e}'
            logger.error(msg)
            return f'<span style="color:red">{msg}</span>'


re_command = re.compile(r'''^!(\w+\(([^)]*)\))''')

def ast_node_to_value(node):
    if isinstance(node, ast.Constant):
        # Handle atomic literals like numbers, strings, etc.
        return node.value
    elif isinstance(node, ast.List):
        # Handle list literals
        return [ast_node_to_value(element) for element in node.elts]
    elif isinstance(node, ast.Tuple):
        # Handle tuple literals
        return tuple(ast_node_to_value(element) for element in node.elts)
    elif isinstance(node, ast.Dict):
        # Handle dict literals
        return {ast_node_to_value(key): ast_node_to_value(value) for key, value in zip(node.keys, node.values)}
    elif isinstance(node, ast.Set):
        # Handle set literals
        return {ast_node_to_value(element) for element in node.elts}
    # Add more cases here for other compound types if needed
    else:
        raise TypeError("Unsupported AST node type")

class CommandProtocol(Protocol):
    '''
    Command protocol

    Handle custom commands
    '''
    protocol_id: str = 'cmd'

    def __init__(self, config:Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    async def arun(self):
        asyncio.create_task(super().arun())

    async def on_ws_chat_input(self, content:str):
        'receive command syntax on the mq chat channel'

        # !gameio_start(game='y93', players=['user1', 'user2'])

        for match in re_command.finditer(content):
            call_str = match.group(1)

            result = await self.send('cmd', call_str)

            if result:
                await self.send('ws', 'append_chat', author='info', content=result)

    async def do_send(self, cmd:str, **kwargs):
        # Parse cmd as a function call expression using ast
        result = ''

        try:
            node = ast.parse(cmd, mode='eval').body

            if isinstance(node, ast.Name):
                return await super().do_send(cmd, **kwargs)

            elif isinstance(node, ast.Call):
                func_name = node.func.id
                kwargs |= {kw.arg: ast_node_to_value(kw.value) for kw in node.keywords}
                result = await self.send('cmd', func_name, **kwargs)
            else:
                result = f'error: Invalid command syntax: {cmd}'

        except (SyntaxError, ValueError) as e:
            # This might occur if the matched string isn't a valid Python function call
            result = f'error: {e} `{cmd}`'

        result = result or ''

        if result.startswith('error'):
            logger.error(result)
        else:
            logger.info('%s => "%s"', cmd, result)

        return result

