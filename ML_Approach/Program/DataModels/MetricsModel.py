from typing import List

from DataModels.Metrics import Metrics


class MetricsModel:
    def __init__(self, metrics_list: List[Metrics], method_name: str):
        self.method_name = method_name
        self.metrics_list = metrics_list
