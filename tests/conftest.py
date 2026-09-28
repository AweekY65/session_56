"""确保测试（以及整个模拟器）不会发起任何网络连接：把 socket 全部打桩为失败。"""
import socket

_real_socket = socket.socket


def _deny_socket(*args, **kwargs):
    raise AssertionError(
        "联邦学习模拟器必须完全本地，禁止创建任何网络 socket（检测到网络连接尝试）"
    )


def pytest_sessionstart(session):  # noqa: ARG001
    socket.socket = _deny_socket  # type: ignore[assignment]


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    socket.socket = _real_socket  # type: ignore[assignment]
