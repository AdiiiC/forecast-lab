from __future__ import annotations
import json
import pandas as pd

class KafkaConnector:
    """Streaming connector. `read()` polls a bounded batch; `write()` produces."""
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "fl"):
        from kafka import KafkaConsumer, KafkaProducer  # type: ignore
        self.topic = topic
        self.consumer = KafkaConsumer(
            topic, bootstrap_servers=bootstrap_servers, group_id=group_id,
            value_deserializer=lambda b: json.loads(b.decode()),
            enable_auto_commit=True, auto_offset_reset="latest")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode())

    def read(self, *, where=None, max_records: int = 1000) -> pd.DataFrame:
        recs = []
        for i, msg in enumerate(self.consumer):
            recs.append(msg.value)
            if i + 1 >= max_records: break
        return pd.DataFrame(recs)

    def write(self, df: pd.DataFrame, mode: str = "append") -> None:
        for r in df.to_dict(orient="records"):
            self.producer.send(self.topic, r)
        self.producer.flush()