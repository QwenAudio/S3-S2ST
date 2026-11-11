
class StateStore(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def set_reuse_state(self):
        self['reuse_state'] = True
    def get_reuse_state(self):
        return self.get('reuse_state', False)
    def get_or_create(self, key):
        if key not in self:
            self[key] = StateStore()
        self[key]['reuse_state'] = self.get_reuse_state()
        return self[key]

# Single ChunkFlowStateStoreManager
class ChunkFlowStateStoreManager:
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance
    def __init__(self):
        self.state_store = None
    def alloc_state_store(self):
        assert self.state_store is None, f"Only one state store is allowed in ChunkFlowStateStoreManager, Maybe you missed calling clear_state_store()"
        self.state_store = StateStore()
    def clear_state_store(self):
        self.state_store = None
    def reset_state_store(self):
        self.clear_state_store()
        self.alloc_state_store()
    def get_state_store(self):
        assert self.state_store is not None, f"State store is not allocated"
        return self.state_store

# Create a singleton state instance
state_store_manager = ChunkFlowStateStoreManager()