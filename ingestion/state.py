import json
import os
import hashlib
from typing import Dict

class IngestionState:
    def __init__(self, state_file: str = "ingestion/state.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_file_changed(self, file_path: str) -> bool:
        current_hash = self.get_file_hash(file_path)
        last_hash = self.state.get(file_path)
        return current_hash != last_hash

    def update_file(self, file_path: str):
        self.state[file_path] = self.get_file_hash(file_path)
