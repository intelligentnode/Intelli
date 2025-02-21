import mmap
import os

class DeepSeekMemoryMapper:
    def __init__(self, model_path: str):
        self.file = open(model_path, "rb")
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
    def get_tensor(self, offset: int, size: int) -> memoryview:
        return self.mmap[offset:offset+size]
    
    def __del__(self):
        self.mmap.close()
        self.file.close() 