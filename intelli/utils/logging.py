class Logger:
    def __init__(self, enable_logging=True, head_size=200):
        self.enable_logging = enable_logging
        self.head_size = head_size

    def log_head(self, message, data=None):
        if self.enable_logging:
            if data:
                print(f"{message}: {data[:self.head_size]}")
            else:
                print(message)

    def log(self, message, data=None):
        if self.enable_logging:
            if data:
                print(f"{message}: {data}")
            else:
                print(message)
