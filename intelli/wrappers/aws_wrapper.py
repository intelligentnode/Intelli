import base64
import datetime
import hashlib
import hmac
import json
import os
import re
import struct
import time
import uuid
import zlib
from urllib.parse import parse_qsl, quote, urlsplit

import requests

from intelli.config import config


class AWSError(Exception):
    """
    Error raised by AWSWrapper.

    It is a plain Exception subclass, so existing `except Exception` handlers keep working.
    API keys and secret keys are removed from the message.
    """

    def __init__(self, message, status_code=None, error_type=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.details = details


class AWSWrapper:
    """
    One wrapper for AWS AI services, called over REST (no AWS SDK needed).

    Amazon Bedrock: Converse (chat, tools, vision, documents, streaming), InvokeModel, embeddings,
    image and video generation, token counting, guardrails, model listing, Knowledge Bases,
    Bedrock Agents and AgentCore Runtime. Amazon Polly: text to speech.

    Auth, in this order:
    - Bedrock API key (bearer token): api_key, or the AWS_BEARER_TOKEN_BEDROCK environment variable.
      It works for Bedrock model calls only (not Knowledge Bases, Agents, AgentCore or Polly).
    - IAM keys, signed with SigV4: access_key_id / secret_access_key (/ session_token), or the
      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables.
    - The AWS SDK credential chain (profiles, SSO, IAM roles): optional, needs `pip install intelli[aws]`.
      It is used when nothing above is given, or when profile / credentials is passed.

    Examples:
        AWSWrapper(bedrock_api_key, region="us-east-1")                              # Bedrock API key
        AWSWrapper(access_key_id=KEY_ID, secret_access_key=SECRET, region="us-east-1")  # IAM keys
        AWSWrapper(profile="dev")                                                    # ~/.aws profile (needs boto3)
        AWSWrapper()                                                                 # environment / default chain

    See instructions/how to docs/AWS_WRAPPER.md for the full guide.
    """

    # Endpoints that accept a Bedrock API key. The others need SigV4.
    _BEARER_SERVICES = ('bedrock-runtime', 'bedrock')
    _SIGNING_NAMES = {
        'bedrock-runtime': 'bedrock',
        'bedrock': 'bedrock',
        'bedrock-agent-runtime': 'bedrock',
        'bedrock-agentcore': 'bedrock-agentcore',
        'polly': 'polly',
    }
    # Errors that move converse() to the next fallback model: access not granted, throttling
    # and quota, timeouts and service errors. A bad request (400) stops the chain.
    _FALLBACK_STATUS = (403, 404, 408, 424, 429, 500, 503)
    # Convenience request keys that converse() moves into inferenceConfig
    _INFERENCE_ALIASES = {'max_tokens': 'maxTokens', 'temperature': 'temperature', 'top_p': 'topP',
                          'stop_sequences': 'stopSequences'}
    _MEDIA_FORMATS = {
        'image': ('png', 'jpeg', 'gif', 'webp'),
        'document': ('pdf', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'txt', 'md'),
        'video': ('mp4', 'mov', 'mkv', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp'),
        'audio': ('mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'opus'),
    }
    # (region, model) -> time until converse() skips the model. Shared by all wrappers in the
    # process, because Chatbot and Agents build a new wrapper for every call.
    _cooldowns = {}

    def __init__(self, api_key=None, timeout=180, *, region=None, access_key_id=None, secret_access_key=None,
                 session_token=None, profile=None, credentials=None, base_url=None, session=None,
                 fallback_cooldown=0):
        aws_cfg = config['url']['aws']
        self.timeout = timeout
        self.profile = profile
        self.fallback_cooldown = fallback_cooldown or 0
        self.last_model = None
        self._credentials = credentials
        self._sdk_session = None
        self._last_secrets = ()

        self._static_credentials = None
        if access_key_id and secret_access_key:
            self._static_credentials = (access_key_id.strip(), secret_access_key.strip(),
                                        session_token.strip() if session_token else None)
        elif access_key_id or secret_access_key:
            raise AWSError("access_key_id and secret_access_key must be given together.")

        api_key = api_key.strip() if isinstance(api_key, str) else api_key
        if not api_key and not (self._static_credentials or credentials is not None or profile):
            api_key = (os.getenv('AWS_BEARER_TOKEN_BEDROCK') or '').strip()
        self.api_key = api_key or None

        self.region = (region or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')
                       or self._sdk_region() or aws_cfg['default_region'])

        if isinstance(base_url, str):
            base_url = {'bedrock-runtime': base_url}
        self._base_urls = {name: url.rstrip('/') for name, url in (base_url or {}).items()}
        self.session = session if session is not None else requests.Session()

    @classmethod
    def from_options(cls, api_key=None, options=None, timeout=None):
        """
        Build a wrapper from a Chatbot / Agent / controller options dict.

        Recognized keys: region, access_key_id, secret_access_key, session_token, profile (each also
        with an aws_ prefix, for example aws_region), credentials, fallback_cooldown, timeout.
        """
        options = options or {}
        kwargs = {}
        for name in ('region', 'access_key_id', 'secret_access_key', 'session_token', 'profile'):
            value = options.get(name) or options.get('aws_' + name)
            if value:
                kwargs[name] = value
        for name in ('credentials', 'fallback_cooldown'):
            if options.get(name) is not None:
                kwargs[name] = options[name]
        if timeout is None:
            timeout = options.get('timeout', 180)
        return cls(api_key, timeout=timeout, **kwargs)

    # ------------------------------------------------------------------
    # Credentials and SigV4
    # ------------------------------------------------------------------
    def _get_sdk_session(self, required=True):
        """The botocore session used for profiles, SSO and IAM roles (optional dependency)."""
        if self._sdk_session is None:
            try:
                import botocore.session
            except ImportError:
                if not required:
                    return None
                raise AWSError("No AWS credentials were given. Pass a Bedrock API key (api_key), or "
                               "access_key_id and secret_access_key, or set AWS_ACCESS_KEY_ID and "
                               "AWS_SECRET_ACCESS_KEY. Profiles, SSO and IAM roles need the AWS SDK: "
                               "pip install intelli[aws]") from None
            self._sdk_session = botocore.session.Session(profile=self.profile)
        return self._sdk_session

    def _sdk_region(self):
        region = getattr(self._credentials, 'region_name', None)
        if region or self.api_key or self._static_credentials:
            return region
        try:
            session = self._get_sdk_session(required=False)
            return session.get_config_variable('region') if session is not None else None
        except Exception:
            return None

    def _resolve_credentials(self):
        """Return (access_key, secret_key, session_token) for SigV4."""
        if self._static_credentials:
            credentials = self._static_credentials
        else:
            source = self._credentials
            if source is None and not self.profile and os.getenv('AWS_ACCESS_KEY_ID') and os.getenv(
                    'AWS_SECRET_ACCESS_KEY'):
                source = {'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
                          'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                          'token': os.getenv('AWS_SESSION_TOKEN')}
            if source is None:
                source = self._get_sdk_session()
            try:
                # boto3 / botocore session -> credentials -> a frozen copy (roles refresh here)
                if callable(getattr(source, 'get_credentials', None)):
                    source = source.get_credentials()
                if callable(getattr(source, 'get_frozen_credentials', None)):
                    source = source.get_frozen_credentials()
            except Exception as error:
                raise AWSError(f"Could not load AWS credentials ({type(error).__name__}: {error})") from None
            if isinstance(source, dict):
                credentials = (source.get('access_key'), source.get('secret_key'), source.get('token'))
            else:
                credentials = (getattr(source, 'access_key', None), getattr(source, 'secret_key', None),
                               getattr(source, 'token', None))
            if not credentials[0] or not credentials[1]:
                raise AWSError("No AWS credentials found. Pass a Bedrock API key (api_key), or access_key_id "
                               "and secret_access_key, or configure a profile (aws configure).")
            credentials = (credentials[0].strip(), credentials[1].strip(),
                           credentials[2].strip() if credentials[2] else None)
        self._last_secrets = (credentials[1], credentials[2])
        return credentials

    @staticmethod
    def _utcnow():
        return datetime.datetime.now(datetime.timezone.utc)

    def _sigv4_headers(self, method, url, body, signing_name, extra_headers=None):
        """AWS Signature Version 4 headers for one request."""
        access_key, secret_key, token = self._resolve_credentials()
        amz_date = self._utcnow().strftime('%Y%m%dT%H%M%SZ')
        date = amz_date[:8]
        parts = urlsplit(url)

        to_sign = {'host': parts.netloc, 'x-amz-date': amz_date}
        if token:
            to_sign['x-amz-security-token'] = token
        for name, value in (extra_headers or {}).items():
            to_sign[name.lower()] = ' '.join(str(value).split())
        names = sorted(to_sign)
        signed_headers = ';'.join(names)

        query = sorted((quote(key, safe='-_.~'), quote(value, safe='-_.~'))
                       for key, value in parse_qsl(parts.query, keep_blank_values=True))
        canonical_request = '\n'.join([
            method,
            # The path is already percent-encoded once; SigV4 encodes it a second time.
            quote(parts.path or '/', safe='/~'),
            '&'.join(f'{key}={value}' for key, value in query),
            ''.join(f'{name}:{to_sign[name]}\n' for name in names),
            signed_headers,
            hashlib.sha256(body or b'').hexdigest(),
        ])
        scope = f'{date}/{self.region}/{signing_name}/aws4_request'
        string_to_sign = '\n'.join(['AWS4-HMAC-SHA256', amz_date, scope,
                                    hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()])
        key = ('AWS4' + secret_key).encode('utf-8')
        for part in (date, self.region, signing_name, 'aws4_request'):
            key = hmac.new(key, part.encode('utf-8'), hashlib.sha256).digest()
        signature = hmac.new(key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

        headers = {
            'X-Amz-Date': amz_date,
            'Authorization': f'AWS4-HMAC-SHA256 Credential={access_key}/{scope}, '
                             f'SignedHeaders={signed_headers}, Signature={signature}',
        }
        if token:
            headers['X-Amz-Security-Token'] = token
        return headers

    # ------------------------------------------------------------------
    # Requests and errors
    # ------------------------------------------------------------------
    def _endpoint(self, service):
        if service in self._base_urls:
            return self._base_urls[service]
        return config['url']['aws']['endpoints'][service].format(region=self.region)

    @staticmethod
    def _path(value):
        """Percent-encode one URL path segment (model ids and ARNs contain ':' and '/')."""
        return quote(str(value), safe='')

    def _redact(self, text):
        text = str(text)
        secrets = [self.api_key, *self._last_secrets]
        if self._static_credentials:
            secrets.extend(self._static_credentials[1:])
        for secret in sorted({s for s in secrets if isinstance(s, str) and len(s) >= 6}, key=len, reverse=True):
            text = text.replace(secret, '<redacted>')
        return text

    def _api_error(self, prefix, response):
        status = response.status_code
        try:
            details = response.json()
        except ValueError:
            details = (response.text or '')[:2000] or None
        error_type = (response.headers.get('x-amzn-errortype') or '').split(':')[0] or None
        message = details
        if isinstance(details, dict):
            message = details.get('message') or details.get('Message')
            if not error_type and details.get('__type'):
                error_type = str(details['__type']).split('#')[-1]
        text = f"{prefix}: HTTP {status}"
        if error_type:
            text += f" {error_type}"
        if message:
            text += f": {message}"
        lowered = text.lower()
        if 'use case' in lowered:
            text += (" (hint: submit the Anthropic use case details form in the Bedrock console, "
                     "under Model access, then retry after a few minutes.)")
        elif 'on-demand throughput' in lowered or 'model identifier is invalid' in lowered:
            text += (" (hint: use the inference profile id of the model, for example "
                     "us.anthropic.claude-sonnet-4-6 instead of anthropic.claude-sonnet-4-6.)")
        return AWSError(self._redact(text), status_code=status, error_type=error_type, details=details)

    def _request(self, method, service, path, body=None, *, query=None, headers=None, data=None,
                 stream=False, bearer=None, error_prefix='AWS API error'):
        url = self._endpoint(service) + path
        pairs = [(key, value) for key, value in (query or {}).items() if value is not None]
        if pairs:
            url += '?' + '&'.join(
                f"{quote(str(key), safe='-_.~')}="
                f"{quote(str(value).lower() if isinstance(value, bool) else str(value), safe='-_.~')}"
                for key, value in pairs)
        if body is not None:
            data = json.dumps(body).encode('utf-8')

        request_headers = {'Content-Type': 'application/json'}
        request_headers.update({key: value for key, value in (headers or {}).items() if value is not None})
        token = bearer or (self.api_key if service in self._BEARER_SERVICES else None)
        if token:
            request_headers['Authorization'] = f'Bearer {token}'
        else:
            try:
                request_headers.update(self._sigv4_headers(method, url, data, self._SIGNING_NAMES[service]))
            except AWSError as error:
                if self.api_key:
                    raise AWSError(f"{error} (a Bedrock API key cannot call {service}; "
                                   "it needs IAM credentials.)") from None
                raise
        try:
            response = self.session.request(method, url, data=data, headers=request_headers,
                                            timeout=self.timeout, stream=stream)
        except requests.exceptions.RequestException as error:
            raise AWSError(self._redact(f"{error_prefix}: {error}")) from None
        if response.status_code >= 400:
            raise self._api_error(error_prefix, response)
        return response

    def _request_json(self, method, service, path, body=None, *, error_prefix='AWS API error', **kwargs):
        response = self._request(method, service, path, body, error_prefix=error_prefix, **kwargs)
        if not (response.content or b'').strip():
            return {}
        try:
            return response.json()
        except ValueError:
            raise AWSError(f"{error_prefix}: the API returned a response that is not JSON") from None

    # ------------------------------------------------------------------
    # Event streams (application/vnd.amazon.eventstream)
    # ------------------------------------------------------------------
    @staticmethod
    def _event_headers(data):
        headers = {}
        position = 0
        # value sizes of the fixed-length header types (true, false, byte, short, int, long, -, -, timestamp, uuid)
        fixed = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}
        while position < len(data):
            name_length = data[position]
            name = data[position + 1:position + 1 + name_length].decode('utf-8')
            position += 1 + name_length
            value_type = data[position]
            position += 1
            if value_type in (6, 7):  # bytes, string
                (length,) = struct.unpack('>H', data[position:position + 2])
                value = data[position + 2:position + 2 + length]
                position += 2 + length
                headers[name] = value.decode('utf-8') if value_type == 7 else value
            elif value_type in fixed:
                value = data[position:position + fixed[value_type]]
                position += fixed[value_type]
                headers[name] = True if value_type == 0 else False if value_type == 1 else value
            else:
                raise AWSError(f"Event stream error: unknown header type {value_type}")
        return headers

    @classmethod
    def _iter_event_stream(cls, response):
        """Decode an AWS event stream into (headers, payload bytes) messages."""
        buffer = b''
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue
            buffer += chunk
            while len(buffer) >= 12:
                total_length, headers_length, prelude_crc = struct.unpack('>III', buffer[:12])
                if zlib.crc32(buffer[:8]) != prelude_crc:
                    raise AWSError("Event stream error: the message prelude is corrupt")
                if len(buffer) < total_length:
                    break
                message, buffer = buffer[:total_length], buffer[total_length:]
                if zlib.crc32(message[:-4]) != struct.unpack('>I', message[-4:])[0]:
                    raise AWSError("Event stream error: the message checksum does not match")
                yield cls._event_headers(message[12:12 + headers_length]), message[12 + headers_length:-4]
        if buffer:
            raise AWSError("Event stream error: the stream ended in the middle of a message")

    def _stream_events(self, response, error_prefix):
        """Yield {event_type: payload} dicts, the same shape boto3 returns for stream events."""
        try:
            for headers, payload in self._iter_event_stream(response):
                try:
                    data = json.loads(payload) if payload else {}
                except ValueError:
                    data = {'raw': payload.decode('utf-8', errors='replace')}
                if headers.get(':message-type') in ('exception', 'error'):
                    error_type = headers.get(':exception-type') or headers.get(':error-code')
                    message = (data.get('message') if isinstance(data, dict) else None) or headers.get(
                        ':error-message') or ''
                    raise AWSError(self._redact(f"{error_prefix}: {error_type}: {message}"),
                                   error_type=error_type, details=data)
                yield {headers.get(':event-type'): data}
        finally:
            response.close()

    # ------------------------------------------------------------------
    # Converse helpers
    # ------------------------------------------------------------------
    def _geo(self):
        """Cross-region inference profile prefix for the wrapper region."""
        region = self.region or ''
        if region.startswith('us-gov-'):
            return 'us-gov'
        return {'eu': 'eu', 'ap': 'apac'}.get(region.split('-')[0], 'us')

    def _default_model(self, kind):
        return config['url']['aws']['models'][kind].format(geo=self._geo())

    @staticmethod
    def _tool_choice(choice):
        if isinstance(choice, str):
            # 'none' has no Bedrock equivalent, so it is left out.
            return {'auto': {'auto': {}}, 'any': {'any': {}}, 'required': {'any': {}}}.get(choice.lower())
        if isinstance(choice, dict):
            if any(key in choice for key in ('auto', 'any', 'tool')):
                return choice
            function = choice.get('function')
            name = function.get('name') if isinstance(function, dict) else choice.get('name')
            if name:
                return {'tool': {'name': name}}
            if choice.get('type') == 'auto':
                return {'auto': {}}
            if choice.get('type') in ('any', 'required'):
                return {'any': {}}
        return None

    @staticmethod
    def to_tool_config(tools, tool_choice=None):
        """
        Convert a tool list to a Converse toolConfig.

        Accepts OpenAI tools ({"type": "function", "function": {...}}), Anthropic tools
        ({"name", "description", "input_schema"}) and Bedrock tools ({"toolSpec": {...}}).
        """
        specs = []
        for tool in tools:
            function = tool.get('function') if isinstance(tool.get('function'), dict) else tool
            if not function.get('name'):
                specs.append(tool)  # already a Bedrock tool (toolSpec, cachePoint, ...)
                continue
            schema = (function.get('parameters') or function.get('input_schema')
                      or {'type': 'object', 'properties': {}})
            spec = {'name': function['name'], 'inputSchema': {'json': schema}}
            if function.get('description'):
                spec['description'] = function['description']
            specs.append({'toolSpec': spec})
        tool_config = {'tools': specs}
        choice = AWSWrapper._tool_choice(tool_choice)
        if choice:
            tool_config['toolChoice'] = choice
        return tool_config

    @staticmethod
    def media_block(source, format=None, *, kind=None, name=None):
        """
        Build an image, document, video or audio content block for Converse.

        source: a file path, raw bytes, a base64 string or an s3:// URI.
        format: for example 'png', 'pdf' or 'mp4'. Taken from the file extension when possible.
        kind: 'image', 'document', 'video' or 'audio'. Taken from the format when possible.
        """
        extension = None
        if isinstance(source, (bytes, bytearray)):
            content = {'bytes': base64.b64encode(bytes(source)).decode('ascii')}
        elif isinstance(source, str) and source.startswith('s3://'):
            content = {'s3Location': {'uri': source}}
            extension = os.path.splitext(source)[1]
        elif isinstance(source, str) and os.path.isfile(source):
            with open(source, 'rb') as file:
                content = {'bytes': base64.b64encode(file.read()).decode('ascii')}
            extension = os.path.splitext(source)[1]
            name = name or os.path.splitext(os.path.basename(source))[0]
        elif isinstance(source, str):
            content = {'bytes': source}
        else:
            raise AWSError("media_block needs a file path, bytes, a base64 string or an s3:// URI.")

        media_format = (format or extension or '').lower().lstrip('.')
        media_format = {'jpg': 'jpeg', '3gp': 'three_gp'}.get(media_format, media_format)
        if not kind:
            kind = next((k for k, formats in AWSWrapper._MEDIA_FORMATS.items() if media_format in formats), None)
        if not kind or not media_format:
            raise AWSError("media_block could not tell the media type. Pass format (for example 'png' "
                           "or 'pdf') and kind ('image', 'document', 'video' or 'audio').")
        block = {'format': media_format, 'source': content}
        if kind == 'document':
            # Bedrock allows letters, digits, single spaces, hyphens, parentheses and square brackets.
            cleaned = re.sub(r'[^A-Za-z0-9\-()\[\]]+', ' ', name or 'document').strip()
            block['name'] = cleaned or 'document'
        return {kind: block}

    @staticmethod
    def build_converse(prompt=None, *, system=None, media=None, messages=None, **params):
        """Build a Converse request from a prompt, optional media and extra request fields."""
        if media is None:
            media = []
        elif not isinstance(media, (list, tuple)):
            media = [media]
        content = [item if isinstance(item, dict) else AWSWrapper.media_block(item) for item in media]
        if prompt:
            content.append({'text': prompt})
        body = dict(params)
        body['messages'] = list(messages or []) + ([{'role': 'user', 'content': content}] if content else [])
        if system:
            body['system'] = system
        return body

    def _prepare_converse(self, params, model=None, kind='chat'):
        """Split a request into (model id, Converse body) and map the convenience keys."""
        body = dict(params or {})
        model_id = model or body.get('model') or body.get('modelId') or self._default_model(kind)
        for key in ('model', 'modelId', 'stream', 'fallback_models'):
            body.pop(key, None)

        inference = dict(body.get('inferenceConfig') or {})
        for alias, name in self._INFERENCE_ALIASES.items():
            value = body.pop(alias, None)
            if value is not None:
                inference.setdefault(name, [value] if name == 'stopSequences' and isinstance(value, str) else value)
        if inference:
            body['inferenceConfig'] = inference

        tools = body.pop('tools', None)
        tool_choice = body.pop('tool_choice', None)
        if tools and 'toolConfig' not in body:
            body['toolConfig'] = self.to_tool_config(tools, tool_choice)

        if isinstance(body.get('system'), str):
            body['system'] = [{'text': body['system']}]
        if not body.get('system'):
            body.pop('system', None)
        return model_id, body

    @staticmethod
    def _content_blocks(response):
        if not isinstance(response, dict):
            return []
        return ((response.get('output') or {}).get('message') or {}).get('content') or []

    @staticmethod
    def extract_text(response):
        """Join the text of a Converse response (reasoning and tool blocks are skipped)."""
        texts = []
        for block in AWSWrapper._content_blocks(response):
            if not isinstance(block, dict):
                continue
            if isinstance(block.get('text'), str):
                texts.append(block['text'])
            for part in (block.get('citationsContent') or {}).get('content') or []:
                if isinstance(part, dict) and isinstance(part.get('text'), str):
                    texts.append(part['text'])
        return ''.join(texts)

    @staticmethod
    def extract_tool_calls(response):
        """Return the tool calls of a Converse response in the OpenAI tool_calls shape."""
        calls = []
        for block in AWSWrapper._content_blocks(response):
            tool_use = block.get('toolUse') if isinstance(block, dict) else None
            if tool_use:
                calls.append({
                    'id': tool_use.get('toolUseId'),
                    'type': 'function',
                    'function': {
                        'name': tool_use.get('name'),
                        'arguments': json.dumps(tool_use.get('input') or {}),
                    },
                })
        return calls

    @staticmethod
    def extract_images(response):
        """Return the base64 images of an image model response."""
        if not isinstance(response, dict):
            return []
        images = [image for image in response.get('images') or [] if isinstance(image, str)]
        images.extend(item['base64'] for item in response.get('artifacts') or []
                      if isinstance(item, dict) and item.get('base64'))
        return images

    # ------------------------------------------------------------------
    # Text: Converse, streaming, token counting
    # ------------------------------------------------------------------
    def _candidates(self, model_id, fallback_models):
        models = [model_id]
        if isinstance(fallback_models, str):
            fallback_models = [fallback_models]
        for model in fallback_models or []:
            if model and model not in models:
                models.append(model)
        if self.fallback_cooldown and len(models) > 1:
            now = time.time()
            ready = [model for model in models if self._cooldowns.get((self.region, model), 0) <= now]
            models = ready or models
        return models

    def converse(self, params, model=None, *, fallback_models=None):
        """
        Call the Bedrock Converse API and return the response dict.

        params is a Converse request body. These extra keys are accepted and mapped for you:
        model, tools / tool_choice (OpenAI, Anthropic or Bedrock format), max_tokens, temperature,
        top_p, stop_sequences, and system as a plain string.

        fallback_models: model ids tried in order when the model before them is not available
        (access not granted, throttled, quota reached, timeout or service error). The id that
        answered is stored in last_model. With fallback_cooldown set on the wrapper, a model
        that failed this way is skipped for that many seconds.
        """
        if fallback_models is None:
            fallback_models = (params or {}).get('fallback_models')
        model_id, body = self._prepare_converse(params, model)
        models = self._candidates(model_id, fallback_models)
        for index, candidate in enumerate(models):
            try:
                response = self._request_json('POST', 'bedrock-runtime',
                                              f"/model/{self._path(candidate)}/converse", body,
                                              error_prefix='Bedrock Converse error')
            except AWSError as error:
                if index + 1 < len(models) and error.status_code in self._FALLBACK_STATUS:
                    if self.fallback_cooldown:
                        self._cooldowns[(self.region, candidate)] = time.time() + self.fallback_cooldown
                    continue
                raise
            self.last_model = candidate
            return response

    def converse_stream(self, params, model=None):
        """Call ConverseStream and yield its events, for example {"contentBlockDelta": {...}}."""
        model_id, body = self._prepare_converse(params, model)
        response = self._request('POST', 'bedrock-runtime', f"/model/{self._path(model_id)}/converse-stream",
                                 body, stream=True, error_prefix='Bedrock ConverseStream error')
        self.last_model = model_id
        yield from self._stream_events(response, 'Bedrock ConverseStream error')

    def generate_text(self, prompt, model=None, *, system=None, media=None, fallback_models=None, **params):
        """Send one prompt (with optional system text and media) and return the answer text."""
        request = self.build_converse(prompt, system=system, media=media, **params)
        return self.extract_text(self.converse(request, model, fallback_models=fallback_models))

    def stream_text(self, prompt, model=None, *, system=None, media=None, **params):
        """Like generate_text, but yields the answer text as it arrives."""
        request = self.build_converse(prompt, system=system, media=media, **params)
        for event in self.converse_stream(request, model):
            text = ((event.get('contentBlockDelta') or {}).get('delta') or {}).get('text')
            if text:
                yield text

    def image_to_text(self, prompt, image_data, extension='png', model=None, *, max_tokens=None):
        """Describe an image (base64 string, bytes or file path) and return the Converse response."""
        request = self.build_converse(prompt, media=[self.media_block(image_data, extension, kind='image')],
                                      max_tokens=max_tokens)
        return self.converse(request, model or self._default_model('vision'))

    def count_tokens(self, params, model=None):
        """Count the input tokens of a Converse request. Returns {"inputTokens": n}."""
        model_id, body = self._prepare_converse(params, model)
        converse = {key: body[key] for key in ('messages', 'system', 'toolConfig', 'additionalModelRequestFields')
                    if key in body}
        return self._request_json('POST', 'bedrock-runtime', f"/model/{self._path(model_id)}/count-tokens",
                                  {'input': {'converse': converse}}, error_prefix='Bedrock CountTokens error')

    # ------------------------------------------------------------------
    # InvokeModel: native model requests, embeddings, images, video
    # ------------------------------------------------------------------
    def invoke_model(self, model, body, *, accept='application/json', content_type='application/json',
                     headers=None):
        """Call InvokeModel with the model's native request body. Returns the parsed JSON response."""
        raw = bytes(body) if isinstance(body, (bytes, bytearray)) else None
        response = self._request('POST', 'bedrock-runtime', f"/model/{self._path(model)}/invoke",
                                 None if raw is not None else body, data=raw,
                                 headers={'Accept': accept, 'Content-Type': content_type, **(headers or {})},
                                 error_prefix='Bedrock InvokeModel error')
        try:
            return response.json()
        except ValueError:
            return response.content

    def invoke_model_stream(self, model, body, *, headers=None):
        """Call InvokeModelWithResponseStream and yield the model's native chunks as dicts."""
        response = self._request('POST', 'bedrock-runtime',
                                 f"/model/{self._path(model)}/invoke-with-response-stream", body,
                                 headers=headers, stream=True, error_prefix='Bedrock InvokeModel stream error')
        for event in self._stream_events(response, 'Bedrock InvokeModel stream error'):
            chunk = (event.get('chunk') or {}).get('bytes')
            if chunk:
                yield json.loads(base64.b64decode(chunk))

    def get_embeddings(self, params):
        """
        Embed texts with Amazon Titan (default), Amazon Nova or Cohere embedding models.

        params: {"texts": [...], "model": optional, plus model options such as "dimensions",
        "normalize" (Titan and Nova) or "input_type" (Cohere)}.
        Returns {"embeddings": [[...], ...], "model": id, "input_tokens": n}.
        """
        params = dict(params or {})
        model = params.pop('model', None) or self._default_model('embed')
        texts = params.pop('texts', None) or params.pop('input', None) or []
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise AWSError("get_embeddings needs 'texts'.")

        def call(body):
            return self.invoke_model(model, body)

        vectors, tokens = [], 0
        if 'cohere.embed' in model:
            params.setdefault('input_type', 'search_document')
            for start in range(0, len(texts), 96):
                embeddings = call({'texts': texts[start:start + 96], **params}).get('embeddings')
                if isinstance(embeddings, dict):
                    embeddings = embeddings.get('float') or next(iter(embeddings.values()), [])
                vectors.extend(embeddings or [])
        elif 'nova' in model:
            single = {'embeddingPurpose': params.pop('embedding_purpose', 'GENERIC_INDEX')}
            if params.get('dimensions'):
                single['embeddingDimension'] = params['dimensions']
            for text in texts:
                data = call({'taskType': 'SINGLE_EMBEDDING', 'singleEmbeddingParams': {
                    **single, 'text': {'truncationMode': 'END', 'value': text}}})
                vectors.append((data.get('embeddings') or [{}])[0].get('embedding'))
        else:
            for text in texts:
                data = call({'inputText': text, **params})
                vectors.append(data.get('embedding'))
                tokens += data.get('inputTextTokenCount') or 0
        return {'embeddings': vectors, 'model': model, 'input_tokens': tokens}

    def generate_image(self, prompt, model=None, *, negative_prompt=None, number_of_images=None, width=None,
                       height=None, seed=None, quality=None, cfg_scale=None, params=None):
        """
        Generate images with Amazon Nova Canvas (default), Titan Image Generator or Stability models.

        params: extra fields of the model's native request body, merged into the request.
        Returns the model response; use extract_images() for the base64 images.
        """
        model = model or self._default_model('image')
        extra = dict(params or {})
        if 'stability.' in model:
            body = {'prompt': prompt}
            if negative_prompt:
                body['negative_prompt'] = negative_prompt
            if seed is not None:
                body['seed'] = seed
            body.update(extra)
        elif extra.get('taskType', 'TEXT_IMAGE') != 'TEXT_IMAGE':
            body = extra  # another Amazon image task (INPAINTING, IMAGE_VARIATION, ...)
        else:
            text_params = {'text': prompt}
            if negative_prompt:
                text_params['negativeText'] = negative_prompt
            image_config = {key: value for key, value in (
                ('numberOfImages', number_of_images), ('width', width), ('height', height), ('seed', seed),
                ('quality', quality), ('cfgScale', cfg_scale)) if value is not None}
            image_config.update(extra.pop('imageGenerationConfig', None) or {})
            body = {'taskType': 'TEXT_IMAGE', 'textToImageParams': text_params, **extra}
            if image_config:
                body['imageGenerationConfig'] = image_config
        return self.invoke_model(model, body)

    def start_async_invoke(self, model, model_input, s3_uri):
        """Start an asynchronous model job that writes its output to S3. Returns {"invocationArn": ...}."""
        body = {'modelId': model, 'modelInput': model_input,
                'outputDataConfig': {'s3OutputDataConfig': {'s3Uri': s3_uri}}}
        return self._request_json('POST', 'bedrock-runtime', '/async-invoke', body,
                                  error_prefix='Bedrock StartAsyncInvoke error')

    def get_async_invoke(self, invocation_arn):
        """Get an asynchronous job. status is InProgress, Completed or Failed."""
        return self._request_json('GET', 'bedrock-runtime', f"/async-invoke/{self._path(invocation_arn)}",
                                  error_prefix='Bedrock GetAsyncInvoke error')

    def wait_for_async_invoke(self, invocation_arn, max_wait_time=900, poll_interval=15):
        """Poll an asynchronous job until it is Completed, and return it."""
        deadline = time.time() + max_wait_time
        while True:
            job = self.get_async_invoke(invocation_arn)
            if job.get('status') == 'Completed':
                return job
            if job.get('status') == 'Failed':
                raise AWSError(f"Bedrock async job failed: {job.get('failureMessage')}", details=job)
            if time.time() >= deadline:
                raise AWSError(f"Bedrock async job did not finish within {max_wait_time} seconds.", details=job)
            time.sleep(poll_interval)

    def generate_video(self, prompt, s3_uri, model=None, *, duration_seconds=6, fps=24, dimension='1280x720',
                       seed=None, image=None, params=None):
        """
        Start an Amazon Nova Reel video job. The video is written to s3_uri (s3://bucket/prefix).

        image: optional start frame (file path, bytes or base64 with params={"image_format": "png"}).
        Returns {"invocationArn": ...}; pass it to wait_for_async_invoke().
        """
        extra = dict(params or {})
        text_params = {'text': prompt}
        if image is not None:
            text_params['images'] = [self.media_block(image, extra.pop('image_format', None), kind='image')['image']]
        video_config = {'durationSeconds': duration_seconds, 'fps': fps, 'dimension': dimension}
        if seed is not None:
            video_config['seed'] = seed
        model_input = {'taskType': 'TEXT_VIDEO', 'textToVideoParams': text_params,
                       'videoGenerationConfig': video_config, **extra}
        return self.start_async_invoke(model or self._default_model('video'), model_input, s3_uri)

    # ------------------------------------------------------------------
    # Guardrails and model listing
    # ------------------------------------------------------------------
    def apply_guardrail(self, guardrail_id, text, *, version='DRAFT', source='INPUT'):
        """Check text against a guardrail. action is NONE or GUARDRAIL_INTERVENED."""
        texts = [text] if isinstance(text, str) else list(text)
        body = {'source': source, 'content': [{'text': {'text': item}} for item in texts]}
        return self._request_json(
            'POST', 'bedrock-runtime',
            f"/guardrail/{self._path(guardrail_id)}/version/{self._path(version)}/apply", body,
            error_prefix='Bedrock ApplyGuardrail error')

    def list_foundation_models(self, provider=None, output_modality=None, inference_type=None):
        """List the foundation models of the region. output_modality: TEXT, IMAGE or EMBEDDING."""
        query = {'byProvider': provider, 'byOutputModality': output_modality, 'byInferenceType': inference_type}
        return self._request_json('GET', 'bedrock', '/foundation-models', query=query,
                                  error_prefix='Bedrock ListFoundationModels error')

    def list_inference_profiles(self, profile_type=None, max_results=None, next_token=None):
        """List inference profiles (the ids such as us.anthropic.claude-sonnet-4-6)."""
        query = {'type': profile_type, 'maxResults': max_results, 'nextToken': next_token}
        return self._request_json('GET', 'bedrock', '/inference-profiles', query=query,
                                  error_prefix='Bedrock ListInferenceProfiles error')

    # ------------------------------------------------------------------
    # Knowledge Bases, Agents and AgentCore (IAM credentials only)
    # ------------------------------------------------------------------
    def retrieve(self, knowledge_base_id, query, number_of_results=None, *, search_type=None, filter=None,
                 next_token=None):
        """Search a Bedrock Knowledge Base. Returns {"retrievalResults": [...]}."""
        vector = {key: value for key, value in (('numberOfResults', number_of_results),
                                                ('overrideSearchType', search_type), ('filter', filter))
                  if value is not None}
        body = {'retrievalQuery': {'text': query}}
        if vector:
            body['retrievalConfiguration'] = {'vectorSearchConfiguration': vector}
        if next_token:
            body['nextToken'] = next_token
        return self._request_json('POST', 'bedrock-agent-runtime',
                                  f"/knowledgebases/{self._path(knowledge_base_id)}/retrieve", body,
                                  error_prefix='Bedrock Retrieve error')

    @staticmethod
    def retrieval_to_text(response):
        """Join the text of the Knowledge Base results."""
        texts = [((result.get('content') or {}).get('text') or '').strip()
                 for result in (response or {}).get('retrievalResults') or []]
        return '\n\n'.join(text for text in texts if text)

    def retrieve_and_generate(self, query, knowledge_base_id, model_arn, *, session_id=None,
                              number_of_results=None):
        """Answer a question from a Knowledge Base. The answer is in output.text, with citations."""
        knowledge_base = {'knowledgeBaseId': knowledge_base_id, 'modelArn': model_arn}
        if number_of_results:
            knowledge_base['retrievalConfiguration'] = {
                'vectorSearchConfiguration': {'numberOfResults': number_of_results}}
        body = {'input': {'text': query},
                'retrieveAndGenerateConfiguration': {'type': 'KNOWLEDGE_BASE',
                                                     'knowledgeBaseConfiguration': knowledge_base}}
        if session_id:
            body['sessionId'] = session_id
        return self._request_json('POST', 'bedrock-agent-runtime', '/retrieveAndGenerate', body,
                                  error_prefix='Bedrock RetrieveAndGenerate error')

    def _agent_events(self, agent_id, agent_alias_id, text, session_id, enable_trace=False, end_session=False,
                      session_state=None):
        body = {'inputText': text}
        if enable_trace:
            body['enableTrace'] = True
        if end_session:
            body['endSession'] = True
        if session_state:
            body['sessionState'] = session_state
        path = (f"/agents/{self._path(agent_id)}/agentAliases/{self._path(agent_alias_id)}"
                f"/sessions/{self._path(session_id)}/text")
        response = self._request('POST', 'bedrock-agent-runtime', path, body, stream=True,
                                 error_prefix='Bedrock Agent error')
        yield from self._stream_events(response, 'Bedrock Agent error')

    def invoke_agent(self, agent_id, agent_alias_id, text, session_id=None, **kwargs):
        """
        Send text to a Bedrock Agent. Pass the returned session_id again to continue the conversation.

        kwargs: enable_trace, end_session, session_state.
        Returns {"text": answer, "session_id": id, "events": [non-text events such as trace]}.
        """
        session_id = session_id or uuid.uuid4().hex
        texts, events = [], []
        for event in self._agent_events(agent_id, agent_alias_id, text, session_id, **kwargs):
            chunk = (event.get('chunk') or {}).get('bytes')
            if chunk:
                texts.append(base64.b64decode(chunk).decode('utf-8', errors='replace'))
            elif 'chunk' not in event:
                events.append(event)
        return {'text': ''.join(texts), 'session_id': session_id, 'events': events}

    def stream_agent(self, agent_id, agent_alias_id, text, session_id=None, **kwargs):
        """Like invoke_agent, but yields the answer text as it arrives."""
        session_id = session_id or uuid.uuid4().hex
        for event in self._agent_events(agent_id, agent_alias_id, text, session_id, **kwargs):
            chunk = (event.get('chunk') or {}).get('bytes')
            if chunk:
                yield base64.b64decode(chunk).decode('utf-8', errors='replace')

    def invoke_agent_runtime(self, agent_runtime_arn, payload, *, session_id=None, qualifier=None,
                             account_id=None, access_token=None, content_type='application/json',
                             accept='application/json', headers=None):
        """
        Invoke an agent hosted on Amazon Bedrock AgentCore Runtime.

        payload: a dict (sent as JSON), a string or bytes, as the agent expects it.
        session_id: at least 33 characters; reuse it to continue a session.
        access_token: an OAuth token, for runtimes with a JWT authorizer (sent instead of SigV4).
        Returns the parsed JSON, the list of events of an SSE response, or the response text.
        """
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
        else:
            data = (payload if isinstance(payload, str) else json.dumps(payload)).encode('utf-8')
        request_headers = {'Content-Type': content_type, 'Accept': accept,
                           'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id': session_id, **(headers or {})}
        response = self._request('POST', 'bedrock-agentcore',
                                 f"/runtimes/{self._path(agent_runtime_arn)}/invocations", data=data,
                                 query={'qualifier': qualifier, 'accountId': account_id},
                                 headers=request_headers, bearer=access_token, error_prefix='AgentCore error')
        if 'text/event-stream' in (response.headers.get('Content-Type') or ''):
            events = []
            for line in response.text.splitlines():
                if line.startswith('data:'):
                    value = line[5:].strip()
                    try:
                        events.append(json.loads(value))
                    except ValueError:
                        events.append(value)
            return events
        try:
            return response.json()
        except ValueError:
            return response.text

    # ------------------------------------------------------------------
    # Amazon Polly: text to speech (IAM credentials only)
    # ------------------------------------------------------------------
    def synthesize_speech(self, text, voice_id=None, *, engine=None, output_format='mp3', language_code=None,
                          text_type=None, sample_rate=None):
        """Convert text to speech with Amazon Polly and return the audio bytes."""
        speech_cfg = config['url']['aws']['speech']
        body = {'Text': text, 'VoiceId': voice_id or speech_cfg['voice'],
                'Engine': engine or speech_cfg['engine'], 'OutputFormat': output_format}
        for key, value in (('LanguageCode', language_code), ('TextType', text_type), ('SampleRate', sample_rate)):
            if value:
                body[key] = str(value)
        return self._request('POST', 'polly', '/v1/speech', body, error_prefix='Polly error').content

    def list_voices(self, engine=None, language_code=None):
        """List the Amazon Polly voices. Returns {"Voices": [...]}."""
        return self._request_json('GET', 'polly', '/v1/voices',
                                  query={'Engine': engine, 'LanguageCode': language_code},
                                  error_prefix='Polly error')
