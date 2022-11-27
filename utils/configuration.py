class Configuration:
    def __init__(self, **config):
        for c_name, c_value in config.items():
            if isinstance(c_value, dict):
                self.__dict__[c_name] = Configuration(**c_value)
            else:
                self.__dict__[c_name] = c_value
