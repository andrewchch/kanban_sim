from simpy import Store, FilterStore


class StoreHistory:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add history tracking
        self.size_history = []
        self.put_queue_size_history = []
        self.get_queue_size_history = []

    def update_history(self):
        """
        Update all tracked histories
        :return:
        """
        self.size_history.append(len(self.items))
        self.put_queue_size_history.append(len(self.put_queue))
        self.get_queue_size_history.append(len(self.get_queue))


class StoreWithHistory(StoreHistory, Store):
    pass


class FilterStoreWithHistory(StoreHistory, FilterStore):
    pass

