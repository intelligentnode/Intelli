class Text2SpeechInput:
    Gender = {
        'FEMALE': 'FEMALE',
        'MALE': 'MALE'
    }

    def __init__(self, text, language="en-gb", gender="FEMALE", voice=None, model='tts-1', stream=True):
        self.text = text
        self.language = language.lower()
        # gender is in uppercase for consistency with Gender dictionary keys
        self.gender = gender.upper()
        self.voice = voice
        self.model = model
        self.stream = stream

    def get_google_input(self):
        params = {'text': self.text, 'languageCode': self.language}

        language_name_map = {
            "en-gb": "en-GB",
            "en": "en-GB",
            "tr-tr": "tr-TR",
            "tr": "tr-TR",
            "cmn-cn": "cmn-CN",
            "cn": "cmn-CN",
            "de-de": "de-DE",
            "de": "de-DE",
            "ar-xa": "ar-XA",
            "ar": "ar-XA",
        }

        gender_name_map = {
            "FEMALE": "A",
            "MALE": "B"
        }

        base_language = language_name_map.get(self.language, None)

        if base_language:
            params['name'] = f"{base_language}-Standard-{gender_name_map.get(self.gender, 'A')}"
            params['ssmlGender'] = self.gender
        else:
            raise ValueError(f"Unsupported language code: {self.language}")

        return params

    def get_openai_input(self):
        return {
            'input': self.text,
            'voice': self.voice,
            'model': self.model,
            'stream': self.stream
        }
