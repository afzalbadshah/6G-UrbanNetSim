class BasePolicy:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.train_mode = False
        self.visualizer = None

    def set_train_mode(self, is_training):
        self.train_mode = bool(is_training)

    def set_visualizer(self, visualizer):
        self.visualizer = visualizer

    def select_action(self, state, mask=None):
        raise NotImplementedError

    def learn(self, transition):
        return None

    def save(self, path):
        return None

    def load(self, path):
        return None
