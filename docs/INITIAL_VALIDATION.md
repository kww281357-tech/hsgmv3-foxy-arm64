# 初始代码快照验证（2026-10-19）

本次任务是将现有 HSGM v3 代码整理为独立私有仓库。没有修改生产导航算法、配置或原实验目录。

## 已通过

- 28 个源码及配置文件复制前后 SHA-256 一致，清单见 SOURCE_SNAPSHOT.json。
- 所有收录 Python 文件通过 AST 语法检查。现有 run_experiments.py 中两个字符串产生 invalid escape sequence 警告，不影响解析。
- 上传范围采用明确文件清单，排除数据、权重、结果、日志、真实 .env、备份和软链接。
- 回归测试仅将旧 edition/HSGM_public_v3/src 路径改为 src；记忆测试中的两个虚构密钥改为显式 test-only 字符串，以免提交检查误报。

## 原有离线回归基线

使用包含 NumPy 的 Python 3.12 环境运行以下四个测试模块，不调用模型 API 或启动仿真实验：

```bash
python scripts/check_repo.py --offline-tests
```

原目录与独立仓库均得到：93 项测试，40 通过、9 失败、43 错误、1 跳过。失败和错误的测试 ID 一致，见 INITIAL_TEST_BASELINE.json。部分测试提取的生产方法缺少新增辅助方法，另有当前楼梯/完成判定与旧测试预期不同的情况。它们在迁移前已经存在，不能把这次上传描述为全部功能验证通过。

test_v3_path_planning_cache.py 未计入轻量回归；它会导入完整 mapper/分割模型，缺少本地缓存时可能自动下载 MobileCLIP 权重。验证过程中发现此行为后已停止该测试，未运行在线导航实验。

没有在全新环境安装全部依赖，没有验证 GPU、Habitat 或真实场景导航效果。后续代码任务应根据改动执行相关检查，并对新增失败负责。

