from .base import Connector
from .parquet import ParquetConnector

CONNECTORS = {"parquet": ParquetConnector}

# Optional connectors registered lazily — only imported if requested.
def get(name: str, **kw) -> Connector:
    if name == "snowflake":
        from .snowflake import SnowflakeConnector
        return SnowflakeConnector(**kw)
    if name == "bigquery":
        from .bigquery import BigQueryConnector
        return BigQueryConnector(**kw)
    if name == "kafka":
        from .kafka import KafkaConnector
        return KafkaConnector(**kw)
    if name in CONNECTORS:
        return CONNECTORS[name](**kw)
    raise ValueError(f"unknown connector: {name}")