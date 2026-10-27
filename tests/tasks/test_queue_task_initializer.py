"""Unit tests for QueueTaskInitializer error handling (no RabbitMQ or database needed)."""
import logging
import multiprocessing

import pytest

from protein_information_system.tasks.queue import QueueTaskInitializer


class DummyTask(QueueTaskInitializer):
    def enqueue(self):
        pass

    def process(self, target):
        return target

    def store_entry(self, record):
        pass


def make_task(**conf):
    """Build a task without running __init__, which connects to RabbitMQ and the DB."""
    task = DummyTask.__new__(DummyTask)
    task.conf = {"rabbitmq_host": "localhost", "rabbitmq_user": "guest", "rabbitmq_password": "guest",
                 "task_retry_delay": 0, **conf}
    task.logger = logging.getLogger("test_queue_task_initializer")
    task.stop_event = multiprocessing.Event()
    task.processes = []
    task.threads = []
    task.setup_rabbitmq = lambda: None
    task.start_workers = lambda: None
    return task


# ---------------------------------------------------------------------------
# Post-processing errors
# ---------------------------------------------------------------------------

def test_post_processing_error_is_raised_when_pipeline_succeeds():
    task = make_task()

    def failing_post_processing():
        raise RuntimeError("summary failed")

    task.post_processing = failing_post_processing

    with pytest.raises(RuntimeError, match="summary failed"):
        task.start()
    assert task.stop_event.is_set()


def test_pipeline_error_is_not_masked_by_post_processing_error():
    task = make_task()

    def failing_workers():
        raise ValueError("worker crashed")

    def failing_post_processing():
        raise RuntimeError("summary failed")

    task.start_workers = failing_workers
    task.post_processing = failing_post_processing

    with pytest.raises(ValueError, match="worker crashed"):
        task.start()


def test_post_processing_runs_after_pipeline_error():
    task = make_task()
    calls = []

    def failing_workers():
        raise ValueError("worker crashed")

    task.start_workers = failing_workers
    task.post_processing = lambda: calls.append("post")

    with pytest.raises(ValueError):
        task.start()
    assert calls == ["post"]


def test_worker_start_errors_reach_start(monkeypatch):
    """Use the real start_workers: a worker that cannot be started must fail start()."""
    import protein_information_system.tasks.queue as queue_module

    class UnstartableProcess:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise OSError("cannot fork")

    task = make_task(max_workers=1)
    del task.start_workers  # back to the class implementation
    task.post_processing = lambda: None
    monkeypatch.setattr(queue_module.multiprocessing, "Process", UnstartableProcess)

    with pytest.raises(OSError, match="cannot fork"):
        task.start()


def test_successful_pipeline_runs_post_processing():
    task = make_task()
    calls = []
    task.post_processing = lambda: calls.append("post")

    task.start()

    assert calls == ["post"]


# ---------------------------------------------------------------------------
# Computing-queue retries
# ---------------------------------------------------------------------------

class FakeChannel:
    def __init__(self):
        self.acks = []
        self.nacks = []
        self.published = []

    def basic_ack(self, delivery_tag):
        self.acks.append(delivery_tag)

    def basic_nack(self, delivery_tag, requeue):
        self.nacks.append((delivery_tag, requeue))

    def basic_publish(self, exchange, routing_key, body):
        self.published.append((routing_key, body))


class Method:
    def __init__(self, delivery_tag=1, redelivered=False):
        self.delivery_tag = delivery_tag
        self.redelivered = redelivered


def _failing_process(target):
    raise RuntimeError("model crashed")


def test_failed_message_is_retried_once():
    import pickle
    task = make_task()
    task.process = _failing_process
    ch = FakeChannel()

    task.callback(ch, Method(redelivered=False), None, pickle.dumps({"x": 1}))

    assert ch.nacks == [(1, True)]


def test_retry_waits_task_retry_delay(monkeypatch):
    import pickle
    import protein_information_system.tasks.queue as queue_module

    slept = []
    monkeypatch.setattr(queue_module.time, "sleep", slept.append)
    task = make_task(task_retry_delay=7)
    task.process = _failing_process
    ch = FakeChannel()

    task.callback(ch, Method(redelivered=False), None, pickle.dumps({"x": 1}))

    assert slept == [7.0]
    assert ch.nacks == [(1, True)]


def test_failed_message_is_discarded_after_retry():
    import pickle
    task = make_task()
    task.process = _failing_process
    ch = FakeChannel()

    task.callback(ch, Method(redelivered=True), None, pickle.dumps({"x": 1}))

    assert ch.nacks == [(1, False)]


def test_successful_message_is_published_and_acked():
    import pickle
    task = make_task()
    ch = FakeChannel()

    task.callback(ch, Method(), None, pickle.dumps({"x": 1}))

    assert ch.acks == [1]
    assert ch.published[0][0] == task.inserting_queue
