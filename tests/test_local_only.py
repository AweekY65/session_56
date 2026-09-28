"""验证模拟器完全本地化：无网络库导入、无 socket 连接、无外部服务调用。

所有数据由代码生成，依赖仅限 numpy 与 Python 标准库。
"""
import ast
import pathlib
import socket

import pytest

import fedavg

PROJECT_ROOT = pathlib.Path(fedavg.__file__).parent

# 显式禁止的网络/外部服务相关模块。
FORBIDDEN_IMPORTS = {
    "requests", "urllib3", "http", "http.client", "urllib.request",
    "aiohttp", "httpx", "grpc", "boto3", "pymongo", "redis",
    "sqlalchemy", "psycopg2", "mysql", "torch.distributed",
    "socket", "ftplib", "telnetlib", "smtplib",
}

ALLOWED_STDLIB = {
    "os", "pickle", "dataclasses", "typing", "numpy", "math",
    "__future__", "abc", "enum", "functools", "itertools", "collections",
    "argparse",
}


def _imported_modules(tree: ast.AST):
    """收集绝对导入的顶层模块；包内相对导入（from .client import ...）跳过。"""
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods


def test_source_has_no_network_imports():
    package_modules = {p.stem for p in PROJECT_ROOT.rglob("*.py")}
    for py in PROJECT_ROOT.rglob("*.py"):
        tree = ast.parse(py.read_text())
        mods = _imported_modules(tree)
        forbidden = mods & FORBIDDEN_IMPORTS
        assert not forbidden, f"{py.name} 包含禁止的外部/网络模块: {forbidden}"
        # 除 fedavg 包内模块、numpy 和白名单标准库外不应有其他第三方依赖。
        third_party = {
            m for m in mods
            if m not in ALLOWED_STDLIB and m not in package_modules
        }
        assert not third_party, f"{py.name} 存在未声明的依赖: {third_party}"


def test_full_run_opens_no_sockets(monkeypatch):
    # 在运行期间把 socket 创建器替换为“必定失败”，确保任何网络尝试都会暴露。
    def _forbidden_socket(*args, **kwargs):
        raise AssertionError("模拟器尝试创建网络 socket，违反完全本地化约束")

    monkeypatch.setattr(socket, "socket", _forbidden_socket)
    # DNS 解析同样禁止。
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("模拟器尝试进行 DNS 解析")
        ),
    )

    from fedavg.simulator import SimConfig, Simulator

    cfg = SimConfig(
        seed=4, rounds=5, n_clients=4, n_samples=800,
        iid=False, alpha=0.3, drop_prob=0.2, fail_prob=0.2,
        participation_frac=0.75,
    )
    sim = Simulator(cfg)
    history = sim.run(verbose=False)
    assert len(history) == 5
    assert history[-1].accuracy > 0.6
    print(f"\n无网络环境下运行成功: final acc={history[-1].accuracy:.4f}")
