class ConnectionException(Exception):
    def __init__(self, code):
        self.message = f'Link not received. Returned http status {code}.'
        self.http_status = code
    def __str__(self):
        return self.message

class GetThreadException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message