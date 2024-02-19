from copy import deepcopy

from intelli.config import config as default_config


class ProxyHelper:
    _instance = None
    API_VERSION = '2023-12-01-preview'

    def __init__(self):
        self.set_default_openai()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ProxyHelper()
        return cls._instance

    def set_default_openai(self):
        config = deepcopy(default_config["url"])
        self.apply_openai_config(config['openai'])

    def set_azure_openai(self, resource_name):
        if not resource_name:
            raise ValueError("Azure resource name must be provided.")

        config = deepcopy(default_config["url"])
        self.resource_name = resource_name
        self.openai_url = config['azure_openai']['base'].replace('{resource-name}', resource_name)
        self.openai_completion = config['azure_openai']['completions']
        self.openai_chat_gpt = config['azure_openai']['chatgpt']
        self.openai_image = config['azure_openai']['imagegenerate']
        self.openai_embed = config['azure_openai']['embeddings']
        self.openai_audio_transcriptions = config['azure_openai']['audiotranscriptions']
        self.openai_audio_speech = config['azure_openai']['audiospeech']
        self.openai_files = config['azure_openai']['files']
        self.openai_finetuning_job = config['azure_openai']['finetuning']
        self.openai_type = 'azure'

    def get_openai_chat_url(self, model=''):
        if self.openai_type == 'azure':
            return self.openai_chat_gpt.replace('{deployment-id}', model).replace('{api-version}',
                                                                                  ProxyHelper.API_VERSION)
        else:
            return self.openai_chat_gpt

    def get_openai_image_url(self):
        """
        Method to get the OpenAI image generation URL
        """
        if self.openai_type == 'azure':
            return self.openai_image.replace('{api-version}', '2023-06-01-preview')
        else:
            return self.openai_image

    def get_openai_embed_url(self, model=''):
        """
        Method to get the Embeddings URL
        """
        if self.openai_type == 'azure':
            return self.openai_embed.replace('{deployment-id}', model).replace('{api-version}', ProxyHelper.API_VERSION)
        else:
            return self.openai_embed

    def get_openai_audio_transcriptions_url(self, model=''):
        """
        Method to get the OpenAI audio transcriptions URL
        """
        if self.openai_type == 'azure':
            return self.openai_audio_transcriptions.replace('{deployment-id}', model).replace('{api-version}',
                                                                                              ProxyHelper.API_VERSION)
        else:
            return self.openai_audio_transcriptions

    def get_openai_audio_speech_url(self, model=''):
        """
        Method to get the OpenAI audio to speech URL
        """
        if self.openai_type == 'azure':
            return self.openai_audio_speech.replace('{deployment-id}', model).replace('{api-version}',
                                                                                      ProxyHelper.API_VERSION)
        else:
            return self.openai_audio_speech

    def get_openai_files_url(self):
        """
        Method to get the OpenAI files endpoint URL
        """
        if self.openai_type == 'azure':
            return self.openai_files.replace('{api-version}', ProxyHelper.API_VERSION)
        else:
            return self.openai_files

    def get_openai_finetuning_job_url(self):
        """
        Method to get the OpenAI fine-tuning job URL
        """
        if self.openai_type == 'azure':
            return self.openai_finetuning_job.replace('{api-version}', '2023-10-01-preview')
        else:
            return self.openai_finetuning_job

    def set_openai_proxy_values(self, proxy_settings):

        self.openai_type = 'custom'

        if proxy_settings and (not proxy_settings['base'] and proxy_settings['url']):
            proxy_settings['base'] = proxy_settings['url']

        adjusted_settings = {
            'base': proxy_settings.get('base', proxy_settings.get('url', self.openai_url)),
            'completions': proxy_settings.get('completions', self.openai_completion),
            'chatgpt': proxy_settings.get('chatgpt', self.openai_chat_gpt),
            'imagegenerate': proxy_settings.get('imagegenerate', self.openai_image),
            'embeddings': proxy_settings.get('embeddings', self.openai_embed),
            'audiotranscriptions': proxy_settings.get('audiotranscriptions', self.openai_audio_transcriptions),
            'audiospeech': proxy_settings.get('audiospeech', self.openai_audio_speech),
            'files': proxy_settings.get('files', self.openai_files),
            'finetuning': proxy_settings.get('finetuning', self.openai_finetuning_job),
            'organization': proxy_settings.get('organization', self.organization),
        }

        self.apply_openai_config(adjusted_settings)

    def apply_openai_config(self, config):

        self.openai_url = config['base']
        self.openai_completion = config['completions']
        self.openai_chat_gpt = config['chatgpt']
        self.openai_image = config['imagegenerate']
        self.openai_embed = config['embeddings']
        self.openai_audio_transcriptions = config['audiotranscriptions']
        self.openai_audio_speech = config['audiospeech']
        self.openai_files = config['files']
        self.openai_finetuning_job = config['finetuning']
        self.openai_type = 'openai'
        self.resource_name = ''
        self.organization = config.get('organization', None)

    # optional getters - to match the right parameter with the provider
    def get_openai_resource_name(self):
        return self.resource_name

    def get_openai_organization(self):
        return self.organization

    def get_openai_type(self):
        return self.openai_type

    def get_openai_url(self):
        return self.openai_url
