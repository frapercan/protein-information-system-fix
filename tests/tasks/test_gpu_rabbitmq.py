"""GPUTaskInitializer RabbitMQ setup: configured port and durable queues (no broker needed)."""
import logging

import protein_information_system.tasks.gpu as gpu_module
from protein_information_system.tasks.gpu import GPUTaskInitializer


class _StopAfterConnect(Exception):
    pass


class DummyGPUTask(GPUTaskInitializer):
    def enqueue(self):
        pass

    def process(self, target):
        return target

    def store_entry(self, record):
        pass


def test_setup_rabbitmq_uses_configured_port(monkeypatch):
    captured = {}

    def fake_blocking_connection(params):
        captured["host"] = params.host
        captured["port"] = params.port
        raise _StopAfterConnect

    monkeypatch.setattr(gpu_module.pika, "BlockingConnection", fake_blocking_connection)

    task = DummyGPUTask.__new__(DummyGPUTask)
    task.conf = {"rabbitmq_host": "localhost", "rabbitmq_port": 24576,
                 "rabbitmq_user": "guest", "rabbitmq_password": "guest",
                 "embedding": {"models": {}}}
    task.logger = logging.getLogger("test_gpu_rabbitmq_port")

    try:
        task.setup_rabbitmq()
    except _StopAfterConnect:
        pass

    assert captured == {"host": "localhost", "port": 24576}


def test_setup_rabbitmq_declares_durable_queues(monkeypatch):
    # RabbitMQ 4.x rejects transient non-exclusive queues
    # (deprecated feature transient_nonexcl_queues), closing the connection.
    declared = []

    class FakeChannel:
        def queue_declare(self, queue, durable=False, **kwargs):
            declared.append((queue, durable))

    class FakeConnection:
        def __init__(self, params):
            pass

        def channel(self):
            return FakeChannel()

    monkeypatch.setattr(gpu_module.pika, "BlockingConnection", FakeConnection)

    task = DummyGPUTask.__new__(DummyGPUTask)
    task.conf = {"rabbitmq_host": "localhost", "rabbitmq_user": "guest", "rabbitmq_password": "guest",
                 "embedding": {"models": {"ESM": {}, "Prot-T5": {}}}}
    task.logger = logging.getLogger("test_gpu_rabbitmq_port")

    task.setup_rabbitmq()

    assert declared and all(durable for _, durable in declared)
