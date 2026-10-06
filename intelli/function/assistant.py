"""
Assistant: a ready-made chat assistant for Gemini- or ChatGPT-style apps on any Intelli Chatbot provider.

    from intelli.function.assistant import Assistant
    from intelli.store import FileChatHistory, MemoryVectorStore

    assistant = Assistant(provider='openai', api_key=key,
                          history=FileChatHistory(dir='./conversations'),
                          knowledge=MemoryVectorStore(embedder={'provider': 'openai', 'api_key': key}))
    assistant.add_documents([{'id': 'handbook', 'text': handbook_text}])
    reply = assistant.chat('What is the refund policy?', conversation_id='c1')
    # reply: {'conversation_id', 'message_id', 'text', 'references', 'citations', 'memories', 'usage', 'model',
    #         'tool_steps'}
"""
import base64
import copy
import inspect
import json
import mimetypes
import os
import re

from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput, ChatMessage
from intelli.store.chat_history import MemoryChatHistory
from intelli.store.vector_store import new_id
from intelli.utils.text_splitter import TextSplitter

DEFAULT_SYSTEM = 'You are a helpful assistant. Answer clearly and concisely.'
GEMINI = ChatProvider.GEMINI.value
ANTHROPIC = ChatProvider.ANTHROPIC.value
OPENAI = ChatProvider.OPENAI.value
AWS = ChatProvider.AWS.value
# providers whose chat API takes OpenAI-style messages and tools
OPENAI_COMPATIBLE = {ChatProvider.MISTRAL.value, ChatProvider.NVIDIA.value, ChatProvider.VLLM.value}
TOOL_PROVIDERS = {OPENAI, ANTHROPIC, GEMINI, AWS} | OPENAI_COMPATIBLE
TEXT_EXTENSIONS = {'txt', 'md', 'markdown', 'csv', 'json', 'html', 'htm', 'xml', 'yaml', 'yml', 'js', 'ts', 'py',
                   'java', 'go', 'rs', 'sql', 'log'}
MIME_BY_EXTENSION = {
    'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'webp': 'image/webp', 'gif': 'image/gif',
    'pdf': 'application/pdf', 'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'ogg': 'audio/ogg', 'm4a': 'audio/mp4',
    'mp4': 'video/mp4', 'mov': 'video/quicktime', 'webm': 'video/webm', 'txt': 'text/plain', 'md': 'text/markdown',
    'csv': 'text/csv', 'html': 'text/html', 'json': 'application/json',
}
# Bedrock document and video formats by mime type
AWS_FORMATS = {'image/jpeg': 'jpeg', 'application/pdf': 'pdf', 'text/plain': 'txt', 'text/markdown': 'md',
               'text/csv': 'csv', 'text/html': 'html', 'video/quicktime': 'mov', 'video/mp4': 'mp4',
               'video/webm': 'webm'}
CITED = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
JSON_TYPES = {str: 'string', int: 'integer', float: 'number', bool: 'boolean', list: 'array', dict: 'object'}


class _AssistantInput(ChatModelInput):
    """
    A ChatModelInput that also carries what an Assistant turn needs on top of plain text: attachments on user
    messages, tool rounds (calls and results), Gemini system instructions and Google Search grounding. The provider
    builders reuse ChatModelInput (model defaults, sampling rules) and replace the messages part.
    """

    def __init__(self, provider, system, tool_definitions=None, google_search=False, **kwargs):
        super().__init__(system, **kwargs)
        self.provider = provider
        self.tool_definitions = tool_definitions or []
        self.google_search = google_search

    def add_user_turn(self, text, attachments=None):
        message = ChatMessage(text, 'user')
        message.attachments = attachments or []
        self.messages.append(message)

    def add_tool_round(self, tool_calls, results, text='', raw=None):
        message = ChatMessage(text or '', 'tool_round')
        message.tool_calls = tool_calls
        message.tool_results = results
        message.raw = raw
        self.messages.append(message)

    def _system_text(self):
        return '\n\n'.join(m.content for m in self.messages if m.role == 'system' and m.content)

    # ---------------- OpenAI chat completions and compatible APIs ----------------
    def _openai_messages(self):
        messages = []
        for m in self.messages:
            if m.role == 'system':
                if m.content:
                    messages.append({'role': 'system', 'content': m.content})
            elif m.role == 'tool_round':
                messages.append({'role': 'assistant', 'content': m.content or None, 'tool_calls': [
                    {'id': call['id'], 'type': 'function',
                     'function': {'name': call['function']['name'], 'arguments': call['function']['arguments']}}
                    for call in m.tool_calls]})
                messages.extend({'role': 'tool', 'tool_call_id': result['id'], 'content': result['content']}
                                for result in m.tool_results)
            elif m.role == 'user' and getattr(m, 'attachments', None):
                parts = [{'type': 'text', 'text': m.content}]
                for item in m.attachments:
                    _require_image(item, self.provider)
                    parts.append({'type': 'image_url', 'image_url': {'url': item.get('uri') or _data_url(item)}})
                messages.append({'role': 'user', 'content': parts})
            else:
                messages.append({'role': m.role, 'content': m.content})
        return messages

    def _with_openai_messages(self, params):
        params['messages'] = self._openai_messages()
        if self.tool_definitions:
            params['tools'] = self.tool_definitions
        return params

    def get_openai_input(self):
        params = super().get_openai_input()
        if self.is_reasoning_model():
            return params  # built by get_openai_gpt5_input
        params.pop('prompt', None)
        return self._with_openai_messages(params)

    def get_mistral_input(self):
        return self._with_openai_messages(super().get_mistral_input())

    def get_nvidia_input(self):
        return self._with_openai_messages(super().get_nvidia_input())

    def get_vllm_input(self):
        params = super().get_vllm_input()
        params.pop('prompt', None)
        return self._with_openai_messages(params)

    # ---------------- OpenAI Responses API (GPT-5 and later) ----------------
    def get_openai_gpt5_input(self):
        params = super().get_openai_gpt5_input()
        items = []
        for m in self.messages:
            if m.role == 'system':
                continue
            if m.role == 'tool_round':
                # the model's own output items (reasoning and function calls) go back as they came
                if m.raw:
                    items.extend(m.raw)
                else:
                    items.extend({'type': 'function_call', 'call_id': call['id'], 'name': call['function']['name'],
                                  'arguments': call['function']['arguments']} for call in m.tool_calls)
                items.extend({'type': 'function_call_output', 'call_id': result['id'], 'output': result['content']}
                             for result in m.tool_results)
            elif m.role == 'user' and getattr(m, 'attachments', None):
                content = [{'type': 'input_text', 'text': m.content}]
                for item in m.attachments:
                    _require_image(item, self.provider)
                    content.append({'type': 'input_image', 'image_url': item.get('uri') or _data_url(item)})
                items.append({'role': 'user', 'content': content})
            else:
                items.append({'role': m.role, 'content': m.content})
        params['input'] = items
        system = self._system_text()
        if system:
            params['instructions'] = system
        if self.tool_definitions:
            params['tools'] = [{'type': 'function', 'name': d['function']['name'],
                                'description': d['function'].get('description', ''),
                                'parameters': d['function']['parameters']} for d in self.tool_definitions]
        return params

    # ---------------- Anthropic ----------------
    def get_anthropic_input(self):
        params = super().get_anthropic_input()
        messages = []
        for m in self.messages:
            if m.role == 'system':
                continue
            if m.role == 'tool_round':
                blocks = [{'type': 'text', 'text': m.content}] if m.content else []
                blocks.extend({'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'],
                               'input': _arguments(call)[0]} for call in m.tool_calls)
                messages.append({'role': 'assistant', 'content': blocks})
                messages.append({'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': result['id'], 'content': result['content'],
                     **({'is_error': True} if result['is_error'] else {})} for result in m.tool_results]})
            elif m.role == 'user' and getattr(m, 'attachments', None):
                blocks = []
                for item in m.attachments:
                    if item.get('uri'):
                        raise ValueError('Anthropic attachments need the file data, not a URI.')
                    mime = item['mime_type']
                    source = {'type': 'base64', 'media_type': mime, 'data': item['data']}
                    if mime == 'application/pdf':
                        blocks.append({'type': 'document', 'source': source})
                    elif mime.startswith('image/'):
                        blocks.append({'type': 'image', 'source': source})
                    else:
                        raise ValueError(f'Anthropic takes image and PDF attachments, not {mime}.')
                blocks.append({'type': 'text', 'text': m.content})
                messages.append({'role': 'user', 'content': blocks})
            else:
                messages.append({'role': m.role, 'content': m.content})
        params['messages'] = messages
        if self.tool_definitions:
            params['tools'] = [{'name': d['function']['name'], 'description': d['function'].get('description', ''),
                                'input_schema': d['function']['parameters']} for d in self.tool_definitions]
        return params

    # ---------------- Gemini (Developer API and Vertex AI) ----------------
    def get_gemini_input(self):
        params = super().get_gemini_input()
        contents = []
        for m in self.messages:
            if m.role == 'system':
                continue
            if m.role == 'tool_round':
                # the model turn goes back as it came, so Gemini keeps its thought signatures
                parts = m.raw or [{'functionCall': {'name': call['function']['name'], 'args': _arguments(call)[0],
                                                    **({'id': call['id']} if call.get('gemini_id') else {})}}
                                  for call in m.tool_calls]
                contents.append({'role': 'model', 'parts': parts})
                contents.append({'role': 'user', 'parts': [{'functionResponse': {
                    'name': result['name'],
                    'response': {'error' if result['is_error'] else 'result': result['content']},
                    **({'id': result['id']} if result.get('gemini_id') else {})}} for result in m.tool_results]})
                continue
            parts = [{'text': m.content}] if m.content else []
            for item in getattr(m, 'attachments', None) or []:
                if item.get('uri'):
                    parts.append({'fileData': {'mimeType': item['mime_type'], 'fileUri': item['uri']}})
                else:
                    parts.append({'inlineData': {'mimeType': item['mime_type'], 'data': item['data']}})
            contents.append({'role': 'model' if m.role == 'assistant' else 'user', 'parts': parts or [{'text': ''}]})
        params['contents'] = contents
        system = self._system_text()
        if system:
            params['systemInstruction'] = {'parts': [{'text': system}]}
        tools = list(params.get('tools') or [])
        if self.tool_definitions:
            tools.append({'functionDeclarations': [
                {'name': d['function']['name'], 'description': d['function'].get('description', ''),
                 'parameters': d['function']['parameters']} for d in self.tool_definitions]})
        if self.google_search:
            tools.append({'googleSearch': {}})
        if tools:
            params['tools'] = tools
        return params

    # ---------------- Amazon Bedrock ----------------
    def get_aws_input(self):
        from intelli.wrappers.aws_wrapper import AWSWrapper
        params = super().get_aws_input()
        messages = []
        for m in self.messages:
            if m.role == 'system':
                continue
            if m.role == 'tool_round':
                content = [{'text': m.content}] if m.content else []
                content.extend({'toolUse': {'toolUseId': call['id'], 'name': call['function']['name'],
                                            'input': _arguments(call)[0]}} for call in m.tool_calls)
                messages.append({'role': 'assistant', 'content': content})
                messages.append({'role': 'user', 'content': [{'toolResult': {
                    'toolUseId': result['id'], 'content': [{'text': result['content'] or ' '}],
                    **({'status': 'error'} if result['is_error'] else {})}} for result in m.tool_results]})
                continue
            content = []
            for item in getattr(m, 'attachments', None) or []:
                mime = item['mime_type']
                kind = 'image' if mime.startswith('image/') else 'video' if mime.startswith('video/') else 'document'
                media_format = AWS_FORMATS.get(mime) or mime.split('/')[-1]
                source = item.get('uri') if str(item.get('uri') or '').startswith('s3://') else item.get('data')
                if not source:
                    raise ValueError('AWS attachments need the file data or an s3:// URI.')
                content.append(AWSWrapper.media_block(source, media_format, kind=kind, name=item.get('name')))
            if m.content:
                content.append({'text': m.content})
            messages.append({'role': m.role, 'content': content or [{'text': ' '}]})
        params['messages'] = messages
        if self.tool_definitions:
            params['tools'] = self.tool_definitions
        return params


class Assistant:
    """
    A ready-made chat assistant for Gemini- or ChatGPT-style apps, on any Intelli Chatbot provider: conversations
    kept in a ChatHistory (memory, JSON files, Firestore), answers grounded on your documents (a knowledge
    VectorStore, with numbered references), long-term memory recalled from earlier conversations (a memory
    VectorStore), attachments (images, PDFs, audio, video on Gemini), Google Search grounding on Gemini / Vertex AI,
    tools, and streaming.
    """

    def __init__(self, provider='openai', api_key=None, model=None, options=None, system_message=DEFAULT_SYSTEM,
                 history=None, knowledge=None, memory=None, max_history=20, top_k=4, memory_top_k=3,
                 min_score=None, google_search=False, tools=None, max_tool_steps=5, max_tokens=None,
                 temperature=None, input_options=None, auto_title=False):
        """
        Args:
            provider: any Chatbot provider: openai, anthropic, gemini, aws, mistral, nvidia, vllm, llamacpp, keras;
                'vertex' is gemini on Vertex AI and 'ollama' is vllm on http://localhost:11434.
            api_key: the provider key (optional for aws with IAM credentials, vllm and local models).
            model: chat model; the provider default when omitted.
            options: Chatbot options: Gemini / Vertex {'vertex', 'project_id', 'location', 'access_token'},
                AWS {'region', ...}, vLLM {'baseUrl'}, {'timeout'}.
            system_message: the assistant's instructions.
            history: conversation store (default MemoryChatHistory).
            knowledge: a VectorStore of documents to ground answers on (RAG).
            memory: a VectorStore for long-term memory: every exchange is stored and recalled later.
            max_history: recent messages sent with each turn (default 20).
            top_k: knowledge chunks per turn (default 4).
            memory_top_k: recalled memories per turn (default 3).
            min_score: drop knowledge and memory matches below this similarity.
            google_search: ground answers on Google Search (gemini and vertex providers).
            tools: functions the model can call: a list of callables, of {'name', 'description', 'parameters',
                'handler'} dicts (or OpenAI {'type': 'function', 'function': {...}, 'handler'}), or {name: callable}.
            max_tool_steps: tool rounds per turn before giving up.
            max_tokens / temperature: sampling settings (provider defaults when omitted).
            input_options: extra ChatModelInput options, sent with every request.
            auto_title: name a new conversation after its first exchange (one extra model call).
        """
        provider = str(provider or OPENAI).lower()
        options = dict(options or {})
        if provider == 'vertex':
            provider = GEMINI
            options.setdefault('vertex', True)
        elif provider == 'ollama':
            provider = ChatProvider.VLLM.value
            options.setdefault('baseUrl', 'http://localhost:11434')
        self.provider = provider
        self.model = model
        self.options = options
        self.chatbot = Chatbot(api_key, provider, options)
        self.system_message = system_message
        self.history = history or MemoryChatHistory()
        self.knowledge = knowledge
        self.memory = memory
        self.max_history = max_history
        self.top_k = top_k
        self.memory_top_k = memory_top_k
        self.min_score = min_score
        self.google_search = google_search
        self.max_tool_steps = max_tool_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.input_options = dict(input_options or {})
        self.auto_title = auto_title
        self.tool_definitions, self.tool_handlers = _tool_registry(tools)
        if self.tool_definitions and provider not in TOOL_PROVIDERS:
            raise ValueError(f'Tools need one of the providers {sorted(TOOL_PROVIDERS)}, not {provider}.')
        if google_search and provider != GEMINI:
            raise ValueError('google_search grounding needs the gemini or vertex provider.')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, message, conversation_id=None, user_id=None, attachments=None, filter=None, system_message=None,
             google_search=None):
        """
        Answer a message in a conversation (a new one when conversation_id is omitted).

        Args:
            attachments: file paths, URLs (http, gs://, s3://), data URLs, {'data' (base64 or bytes), 'mime_type',
                'name'} or {'uri', 'mime_type'}.
            filter: knowledge metadata filter for this turn.
            google_search: Google Search grounding for this turn only (gemini and vertex).

        Returns:
            {'conversation_id', 'message_id', 'text', 'references', 'citations', 'memories', 'usage', 'model',
            'tool_steps'}. references are the retrieved chunks; 'cited': True marks the ones the answer cites as [n].
        """
        turn = self._prepare(message, conversation_id, user_id, attachments, filter, system_message, google_search)
        bot = self._turn_chatbot()
        text, tool_steps = self._run(bot, turn['input'])
        return self._finish(turn, text, bot, tool_steps)

    def stream(self, message, conversation_id=None, user_id=None, attachments=None, filter=None,
               system_message=None, google_search=None):
        """
        Stream the answer. Yields {'type': 'start', 'conversation_id', 'references', 'memories'}, then
        {'type': 'text', 'text'} chunks, then {'type': 'done', ...} with the same fields as chat().
        """
        turn = self._prepare(message, conversation_id, user_id, attachments, filter, system_message, google_search)
        yield {'type': 'start', 'conversation_id': turn['conversation_id'], 'references': turn['references'],
               'memories': turn['memories']}
        bot = self._turn_chatbot()
        text, tool_steps = '', []
        if self.tool_definitions:
            # the tool loop needs whole replies; the final answer is sent as one chunk
            text, tool_steps = self._run(bot, turn['input'])
            if text:
                yield {'type': 'text', 'text': text}
        else:
            try:
                chunks = bot.stream(turn['input'])
                for chunk in chunks:
                    text += chunk
                    yield {'type': 'text', 'text': chunk}
            except NotImplementedError:
                # providers or models without streaming (e.g. GPT-5 on the Responses API): one chunk
                text, _ = self._run(bot, turn['input'])
                if text:
                    yield {'type': 'text', 'text': text}
        yield {'type': 'done', **self._finish(turn, text, bot, tool_steps)}

    def regenerate(self, conversation_id, **options):
        """Answer the last user message of a conversation again (the previous answer is removed)."""
        messages = self.history.get_messages(conversation_id, limit=2)
        if len(messages) < 2 or messages[1]['role'] != 'assistant' or messages[0]['role'] != 'user':
            raise ValueError('The conversation does not end with a user message and an answer.')
        self.history.delete_last_messages(conversation_id, 2)
        if self.memory:
            try:
                self.memory.delete([messages[0]['id']])
            except Exception:
                pass
        return self.chat(messages[0]['content'], conversation_id=conversation_id, **options)

    def add_documents(self, documents, chunk_size=1200, chunk_overlap=150):
        """
        Add documents to the knowledge store, split into chunks with ids '<id>#0', '<id>#1', ...

        Args:
            documents: strings or {'id'?, 'text', 'metadata'?}; metadata source / title / url show in references.

        Returns:
            the chunk ids.
        """
        if not self.knowledge:
            raise ValueError('add_documents needs a knowledge VectorStore: Assistant(knowledge=...).')
        chunks = []
        for document in documents or []:
            item = {'text': document} if isinstance(document, str) else document
            document_id = item.get('id') or new_id()
            metadata = {'source': document_id, **(item.get('metadata') or {})}
            chunks.extend(TextSplitter.to_documents(item.get('text'), metadata, chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap, id_prefix=str(document_id)))
        return self.knowledge.add_documents(chunks)

    def add_files(self, paths, **options):
        """Add text files (txt, md, csv, json, html, code) to the knowledge store; the file name is the source."""
        documents = []
        for path in [paths] if isinstance(paths, str) else paths:
            extension = os.path.splitext(path)[1].lstrip('.').lower()
            name = os.path.basename(path)
            if extension not in TEXT_EXTENSIONS:
                raise ValueError(f'add_files reads text files; extract the text of {name} first (for a PDF on '
                                 'Gemini: GoogleAIWrapper.media_to_text).')
            with open(path, 'r', encoding='utf-8') as file:
                documents.append({'id': name, 'text': file.read(), 'metadata': {'source': name, 'path': path}})
        return self.add_documents(documents, **options)

    def list_conversations(self, user_id=None, limit=50):
        return self.history.list_conversations(user_id=user_id, limit=limit)

    def get_messages(self, conversation_id, limit=None):
        return self.history.get_messages(conversation_id, limit=limit)

    def delete_conversation(self, conversation_id):
        """Delete a conversation and its long-term memories."""
        if self.memory:
            ids = [m['id'] for m in self.history.get_messages(conversation_id) if m['role'] == 'user']
            if ids:
                try:
                    self.memory.delete(ids)
                except Exception:
                    pass
        return self.history.delete_conversation(conversation_id)

    def rename_conversation(self, conversation_id, title):
        return self.history.save_conversation({'id': conversation_id, 'title': title})

    def generate_title(self, conversation_id):
        """Name a conversation from its first message (a short model call) and save the title."""
        first = next((m for m in self.history.get_messages(conversation_id) if m['role'] == 'user'), None)
        if not first:
            return None
        chat_input = self._create_input('You write short titles for chat conversations.', tools=False,
                                        google_search=False, max_tokens=False)
        chat_input.add_user_turn('Write a title of at most six words for a conversation that starts with the '
                                 'message below. Reply with the title only, no quotes.\n\n'
                                 f"{first['content'][:2000]}")
        text, _ = self._run(self._turn_chatbot(), chat_input, tools=False)
        title = re.sub(r'''^["'#\s]+|["'\s]+$''', '', str(text or '')).split('\n')[0][:80] or None
        if title:
            self.history.save_conversation({'id': conversation_id, 'title': title})
        return title

    # ------------------------------------------------------------------
    # Turn building
    # ------------------------------------------------------------------
    def _turn_chatbot(self):
        """A per-turn copy of the shared Chatbot (same wrapper), so concurrent turns keep their own last_response."""
        bot = copy.copy(self.chatbot)
        bot.last_response = None
        return bot

    def _create_input(self, system_text, tools=True, google_search=None, max_tokens=True):
        options = dict(self.input_options)
        temperature = self.temperature
        if temperature is None and self.provider == ChatProvider.LLAMACPP.value:
            temperature = 0.7
        return _AssistantInput(
            self.provider, system_text,
            tool_definitions=self.tool_definitions if tools else None,
            google_search=self.google_search if google_search is None else google_search,
            model=self.model, temperature=temperature,
            max_tokens=self.max_tokens if max_tokens else None, **options)

    def _prepare(self, message, conversation_id, user_id, attachments, filter, system_message, google_search):
        text = '' if message is None else str(message)
        conversation_id = conversation_id or new_id()
        existing = self.history.get_conversation(conversation_id)
        if existing and existing.get('user_id') and user_id and existing['user_id'] != user_id:
            raise PermissionError(f"Conversation '{conversation_id}' belongs to another user.")
        if not existing or (user_id and not existing.get('user_id')):
            fields = {'id': conversation_id}
            if user_id:
                fields['user_id'] = user_id
            self.history.save_conversation(fields)
        owner = user_id or (existing or {}).get('user_id')
        recent = self.history.get_messages(conversation_id, limit=self.max_history) if self.max_history > 0 else []
        references = self._search_knowledge(text, filter)
        memories = self._recall(text, owner, recent)
        files = [_read_attachment(item) for item in attachments or []]
        if google_search and self.provider != GEMINI:
            raise ValueError('google_search grounding needs the gemini or vertex provider.')
        chat_input = self._create_input(self._system_text(system_message or self.system_message, references, memories),
                                        google_search=self.google_search if google_search is None else bool(google_search))
        for item in recent:
            if not item.get('content'):
                continue
            if item['role'] == 'user':
                chat_input.add_user_message(item['content'])
            else:
                chat_input.add_assistant_message(item['content'])
        chat_input.add_user_turn(text, files)
        return {'conversation_id': conversation_id, 'is_new': not existing, 'text': text, 'attachments': files,
                'references': references, 'memories': memories, 'input': chat_input, 'user_id': owner}

    @staticmethod
    def _system_text(system_message, references, memories):
        sections = [system_message]
        if references:
            sections.append('Use the sources below when they are relevant to the question. Cite them inline as [1], '
                            '[2] by their numbers. If the sources do not answer the question, say so before answering '
                            'from general knowledge.')
            sections.append('Sources:\n' + '\n\n'.join(
                f"[{reference['index']}] {Assistant._source_label(reference)}\n{reference['text']}"
                for reference in references))
        if memories:
            sections.append('Notes from earlier conversations with this user (use them only when they help):\n' +
                            '\n'.join('- ' + re.sub(r'\s+', ' ', memory['text'])[:600] for memory in memories))
        return '\n\n'.join(section for section in sections if section)

    @staticmethod
    def _source_label(reference):
        metadata = reference.get('metadata') or {}
        label = metadata.get('title') or metadata.get('source') or reference['id']
        return f"{label} ({metadata['url']})" if metadata.get('url') else str(label)

    def _search_knowledge(self, text, filter):
        if not self.knowledge or not text.strip() or not self.top_k:
            return []
        matches = self.knowledge.query(text=text, top_k=self.top_k, filter=filter)
        return [{'index': index + 1, 'id': match['id'], 'text': match.get('text') or '', 'score': match.get('score'),
                 'metadata': match.get('metadata') or {}}
                for index, match in enumerate(m for m in matches if self._relevant(m))]

    def _recall(self, text, user_id, recent):
        if not self.memory or not text.strip() or not self.memory_top_k:
            return []
        recent_ids = {message['id'] for message in recent}
        matches = self.memory.query(text=text, top_k=self.memory_top_k + len(recent),
                                    filter={'userId': user_id} if user_id else None)
        kept = [m for m in matches if m['id'] not in recent_ids and self._relevant(m)][:self.memory_top_k]
        return [{'id': m['id'], 'text': m.get('text') or '', 'score': m.get('score'),
                 'conversation_id': (m.get('metadata') or {}).get('conversationId')} for m in kept]

    def _relevant(self, match):
        """A match passes min_score; stores that return no score always pass."""
        score = match.get('score')
        return self.min_score is None or not isinstance(score, (int, float)) or score >= self.min_score

    # ------------------------------------------------------------------
    # Model calls and the tool loop
    # ------------------------------------------------------------------
    def _run(self, bot, chat_input, tools=True):
        """Call the model, running tool calls until it answers. Returns (text, tool_steps)."""
        steps = []
        for step in range(self.max_tool_steps + 1):
            replies = bot.chat(chat_input)
            calls = self._tool_calls(replies, bot.last_response) if tools and self.tool_handlers else []
            text = _reply_text(replies)
            if not calls:
                return text, steps
            if step >= self.max_tool_steps:
                break
            results = []
            for call in calls:
                name = call['function']['name']
                args, invalid = _arguments(call)
                is_error = False
                if invalid is not None:
                    # never run a tool with arguments the model did not actually send; let it retry
                    content, is_error = f'Error: the arguments are not valid JSON: {invalid}', True
                else:
                    try:
                        handler = self.tool_handlers.get(name)
                        if not handler:
                            raise ValueError(f"Unknown tool '{name}'.")
                        content = handler(**args)
                    except Exception as error:
                        content, is_error = f'Error: {error}', True
                if not isinstance(content, str):
                    content = json.dumps(content, default=str)
                steps.append({'name': name, 'arguments': args, 'result': content, 'is_error': is_error})
                results.append({'id': call['id'], 'name': name, 'content': content, 'is_error': is_error,
                                'gemini_id': call.get('gemini_id')})
            chat_input.add_tool_round(calls, results, text, self._raw_turn(bot.last_response))
        raise RuntimeError(f'The tool loop stopped after {self.max_tool_steps} rounds without a final answer; '
                           'raise max_tool_steps.')

    def _tool_calls(self, replies, raw):
        """Tool calls in the OpenAI shape [{'id', 'type': 'function', 'function': {'name', 'arguments'}}]."""
        first = replies[0] if isinstance(replies, list) and replies else replies
        if isinstance(first, dict) and first.get('tool_calls'):
            return first['tool_calls']
        if not isinstance(raw, dict):
            return []
        if self.provider == GEMINI:
            from intelli.wrappers.googleai_wrapper import GoogleAIWrapper
            calls = []
            for index, call in enumerate(GoogleAIWrapper.extract_function_calls(raw)):
                calls.append({'id': call.get('id') or f"call_{index}_{call.get('name')}", 'type': 'function',
                              'gemini_id': bool(call.get('id')),
                              'function': {'name': call.get('name'), 'arguments': json.dumps(call.get('args') or {})}})
            return calls
        choices = raw.get('choices') or []
        message = (choices[0] or {}).get('message') or {} if choices else {}
        return message.get('tool_calls') or []

    def _raw_turn(self, raw):
        """The model's own turn to send back with the tool results (Gemini parts, Responses API output items)."""
        if not isinstance(raw, dict):
            return None
        if self.provider == GEMINI:
            candidates = raw.get('candidates') or []
            parts = ((candidates[0] or {}).get('content') or {}).get('parts') if candidates else None
            unalias = getattr(self.chatbot.wrapper, '_unalias', None)
            return unalias(parts) if parts and unalias else parts
        if self.provider == OPENAI and isinstance(raw.get('output'), list):
            return [item for item in raw['output'] if isinstance(item, dict) and item.get('type') != 'message']
        return None

    def _finish(self, turn, text, bot, tool_steps):
        raw = bot.last_response
        citations = _citations(raw) if self.provider == GEMINI else []
        cited = {int(number) for match in CITED.finditer(str(text or '')) for number in match.group(1).split(',')}
        for reference in turn['references']:
            reference['cited'] = reference['index'] in cited
        assistant_metadata = {}
        if turn['references']:
            assistant_metadata['references'] = [{
                'index': r['index'], 'id': r['id'], 'cited': r['cited'], 'source': r['metadata'].get('source'),
                'title': r['metadata'].get('title'), 'url': r['metadata'].get('url')} for r in turn['references']]
        if citations:
            assistant_metadata['citations'] = citations
        user_metadata = {'attachments': [{'name': item.get('name'), 'mime_type': item['mime_type']}
                                         for item in turn['attachments']]} if turn['attachments'] else None
        user_message, assistant_message = self.history.add_messages(turn['conversation_id'], [
            {'role': 'user', 'content': turn['text'], 'metadata': user_metadata},
            {'role': 'assistant', 'content': text, 'metadata': assistant_metadata or None},
        ])
        if self.memory and text:
            metadata = {'conversationId': turn['conversation_id'], 'createdAt': user_message['created_at']}
            if turn['user_id']:
                metadata['userId'] = turn['user_id']
            self.memory.add_documents([{'id': user_message['id'], 'text': f"User: {turn['text']}\nAssistant: {text}",
                                        'metadata': metadata}])
        if self.auto_title and turn['is_new']:
            try:
                self.generate_title(turn['conversation_id'])
            except Exception:
                pass
        return {
            'conversation_id': turn['conversation_id'],
            'message_id': assistant_message['id'],
            'text': text,
            'references': turn['references'],
            'citations': citations,
            'memories': turn['memories'],
            'usage': _usage(raw),
            'model': self._model(raw, bot, turn['input']),
            'tool_steps': tool_steps,
        }

    def _model(self, raw, bot, chat_input):
        """The model that answered: the response's own record when it has one, else the request's model."""
        if self.provider == AWS and getattr(bot.wrapper, 'last_model', None):
            return bot.wrapper.last_model
        if isinstance(raw, dict):
            if raw.get('modelVersion'):
                return raw['modelVersion']
            if isinstance(raw.get('model'), str):
                return raw['model']
        return chat_input.model


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _reply_text(replies):
    first = replies[0] if isinstance(replies, list) and replies else replies
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return first.get('content') or ''
    return ''


def _arguments(call):
    """(args, None) for valid or empty arguments, ({}, raw) when the JSON cannot be parsed."""
    raw = (call.get('function') or {}).get('arguments', call.get('arguments'))
    if isinstance(raw, dict):
        return raw, None
    if raw is None or not str(raw).strip():
        return {}, None
    try:
        args = json.loads(raw)
    except ValueError:
        return {}, str(raw)
    return (args, None) if isinstance(args, dict) else ({}, str(raw))


def _usage(raw):
    """Token usage in one shape for every provider: {'input_tokens', 'output_tokens', 'total_tokens'}."""
    if not isinstance(raw, dict):
        return None
    usage = raw.get('usageMetadata') or raw.get('usage_metadata') or raw.get('usage') or \
        (raw.get('meta') or {}).get('billed_units')
    if not isinstance(usage, dict):
        return None

    def pick(*keys):
        return next((usage[key] for key in keys if usage.get(key) is not None), None)

    input_tokens = pick('promptTokenCount', 'input_tokens', 'prompt_tokens', 'inputTokens')
    output_tokens = pick('candidatesTokenCount', 'output_tokens', 'completion_tokens', 'outputTokens')
    total = pick('totalTokenCount', 'total_tokens', 'totalTokens')
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total}


def _citations(raw):
    """Web sources of a Gemini answer grounded on Google Search: [{'title', 'uri'}]."""
    candidates = (raw or {}).get('candidates') or []
    if not candidates:
        return []
    grounding = (candidates[0] or {}).get('groundingMetadata') or (candidates[0] or {}).get('grounding_metadata') or {}
    citations, seen = [], set()
    for chunk in grounding.get('groundingChunks') or grounding.get('grounding_chunks') or []:
        web = (chunk or {}).get('web') or (chunk or {}).get('retrievedContext') or {}
        uri = web.get('uri')
        if uri and uri not in seen:
            seen.add(uri)
            citations.append({'title': web.get('title'), 'uri': uri})
    return citations


def _data_url(item):
    return f"data:{item['mime_type']};base64,{item['data']}"


def _require_image(item, provider):
    if not str(item.get('mime_type', '')).startswith('image/'):
        raise ValueError(f"{provider} takes image attachments; {item.get('mime_type')} needs the gemini, vertex, "
                         'anthropic or aws provider.')


def _read_attachment(attachment):
    """{'data' (base64), 'mime_type', 'name'} or {'uri', 'mime_type', 'name'} from a path, URL, data URL or dict."""
    if isinstance(attachment, str):
        if re.match(r'^(gs|s3|https?)://', attachment, re.I):
            extension = attachment.split('?')[0].rsplit('.', 1)[-1].lower()
            return {'uri': attachment, 'mime_type': MIME_BY_EXTENSION.get(extension, 'application/octet-stream'),
                    'name': attachment.split('?')[0].split('/')[-1]}
        if attachment.startswith('data:'):
            match = re.match(r'^data:([^;,]+);base64,(.*)$', attachment, re.S)
            if not match:
                raise ValueError('Attachments as data URLs must be base64.')
            return {'data': match.group(2), 'mime_type': match.group(1), 'name': 'attachment'}
        extension = os.path.splitext(attachment)[1].lstrip('.').lower()
        with open(attachment, 'rb') as file:
            data = base64.b64encode(file.read()).decode('ascii')
        mime = MIME_BY_EXTENSION.get(extension) or mimetypes.guess_type(attachment)[0] or 'application/octet-stream'
        return {'data': data, 'mime_type': mime, 'name': os.path.basename(attachment)}
    if isinstance(attachment, dict):
        mime = attachment.get('mime_type') or attachment.get('mimeType')
        uri = attachment.get('uri') or attachment.get('file_uri') or attachment.get('fileUri')
        if uri:
            return {'uri': uri, 'mime_type': mime, 'name': attachment.get('name')}
        if attachment.get('data') is not None:
            if not mime:
                raise ValueError('An attachment with data needs a mime_type.')
            data = attachment['data']
            if isinstance(data, (bytes, bytearray)):
                data = base64.b64encode(bytes(data)).decode('ascii')
            else:
                data = re.sub(r'^data:[^,]*,', '', str(data))
            return {'data': data, 'mime_type': mime, 'name': attachment.get('name')}
    raise ValueError("An attachment is a file path, a URL, a data URL, {'data', 'mime_type'} or {'uri', 'mime_type'}.")


def _tool_registry(tools):
    """Normalise the accepted tool shapes into (OpenAI-style definitions, {name: handler})."""
    definitions, handlers = [], {}
    if not tools:
        return definitions, handlers
    if isinstance(tools, dict):
        tools = [_function_tool(handler, name) for name, handler in tools.items()]
    for tool in tools:
        if callable(tool) and not isinstance(tool, dict):
            tool = _function_tool(tool)
        spec = tool.get('function') if isinstance(tool.get('function'), dict) else tool
        handler = tool.get('handler') or spec.get('handler')
        name = spec.get('name')
        if not name:
            raise ValueError('Every tool needs a name.')
        if callable(handler):
            handlers[name] = handler
        definition = {'name': name, 'parameters': spec.get('parameters') or spec.get('input_schema') or
                      {'type': 'object', 'properties': {}}}
        if spec.get('description'):
            definition['description'] = spec['description']
        definitions.append({'type': 'function', 'function': definition})
    return definitions, handlers


def _function_tool(function, name=None):
    """A tool definition from a Python function: its name, docstring and parameters (type hints map to JSON types)."""
    properties, required = {}, []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = parameter.annotation
        properties[parameter.name] = {'type': JSON_TYPES.get(annotation, 'string')}
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
    parameters = {'type': 'object', 'properties': properties}
    if required:
        parameters['required'] = required
    description = inspect.getdoc(function) or ''
    return {'name': name or function.__name__, 'description': description.split('\n\n')[0], 'parameters': parameters,
            'handler': function}
