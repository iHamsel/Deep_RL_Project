from dataclasses import dataclass


@dataclass
class TrainingEpisodeLogEntry:
   reward:        float
   avgLoss:       float
   playSteps:     int
   learnSteps:    int
   targetUpdates: int
