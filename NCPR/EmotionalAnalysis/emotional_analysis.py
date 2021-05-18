from aip import AipNlp
from EmotionalAnalysis.translate import translate

class EmotionalAnalysis():
    def __init__(self):
        self.app_id = '23895645'
        self.api_key = '0pRnQGCBZbHhPESRIkLMm051'
        self.secret_key = 'iCKFrcZQojr5f5D5cG8Up6xIIUe3Wx7z'
        self.client = AipNlp(self.app_id, self.api_key, self.secret_key)

    def get_score(self, content):
        return self.client.sentimentClassify(content)

if __name__ == '__main__':
    model = EmotionalAnalysis()
    res = model.get_score(translate('yes'))
    print(res)





