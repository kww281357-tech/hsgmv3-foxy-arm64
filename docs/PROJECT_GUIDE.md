# Go2 项目代码入口与验证矩阵

## 分支与来源

- 完整整理入口：`feature/go2-simulation`（本分支）。
- 先前部署基础：`feature/foxy-arm64`；不覆盖该分支。
- `main` 未合并，不代表完整交付；不自动创建 PR 或合并。
- 原 v3 来源：`jkffj54/hsgmv3` @ `3a0a6558a5a485bca7e0793461b9822060b392eb`；原仓库不修改。
- 保留 Apache-2.0 原许可证。外部 Go2 仿真模型使用其 BSD-3-Clause 许可证，不把模型资产/数据集/权重复制入仓库。

## 先验证，再部署

在独立 Python 3.8 环境运行轻量检查（NumPy 等依赖见各复现文档）：

```sh
python scripts/check_repo.py --offline-tests
python tests/test_python38_compat.py
python tests/test_foxy_container.py
python tests/test_monitor_lifecycle.py
```

这不启动完整 Habitat、下载模型或连接机器人。完整 mapper 依赖测试仍需另外准备环境，不能把轻量测试通过说成完整导航通过。

## 工具清单

| 入口 | 要求 | 是否联网/控制机器人 |
|---|---|---|
| `scripts/go2_physics_smoke.py` | Python3.8、MuJoCo3.1.6、外部固定版本 Go2 模型 | 进程内物理，不导入 ROS/SDK，不连接机器人 |
| `scripts/foxy_serialization_check.py` | 真实 Foxy 与编译的 `unitree_go` | 不创建 DDS 节点，仅真实 CDR 消息往返 |
| `scripts/foxy_sim_replay.py` | 隔离的模拟器环境及显式确认参数 | 只回环域211模拟遥测，无运动命令；成功路径未运行 |
| `scripts/check_jetson_host.sh` | 在用户确认的 Jetson 环境运行 | 只读检查，不刷机、不改驱动 |
| `Dockerfile` / `compose.yaml` | 可用 ARM64 Docker 构建器 | 默认回环，只读遥测；尚未构建验证 |

详细参数、模型下载与依赖固定版本见 [物理仿真](GO2_SIMULATION.md)、[Foxy 消息](FOXY_SIMULATION_RUNTIME.md)、[部署与网络](FOXY_ARM64_GO2.md)。

## 已验证与未验证（2026-10-25）

| 层级 | 结果 | 边界 |
|---|---|---|
| 真 CPython3.8 离线回归 | 650/650 通过 | 受保护轻量套件，不等同所有模型测试 |
| 兼容 / 包装 / 生命周期 | 4+4+4 通过 | 包括陈旧遥测与健康检查故障情况 |
| Go2 CPU MuJoCo | 2×5秒、500Hz、完整状态重置一致 | 重力/接触/站立烟雾测试，不是行走或导航 |
| 适配器数据转换 | 250物理样本及 NaN 拒绝通过 | 纯函数层，不是 ROS 收发 |
| Foxy CDR 真消息 | 250条往返通过，每条236字节 | x86_64，档案 Cyclone0.7.0，不是容器0.10.2 |
| DDS 发布订阅 | 阻塞 | 云端接口枚举限制；未绕过，未报通过 |
| ARM64 Docker / Jetson CUDA | 未运行 | 没有镜像产物，需匹配实际 L4T |
| 真 Go2 / 完整自主导航 | 未运行 | 不允许据此上电运动或无人监管运行 |

仓库只保存源码、配置、依赖说明与文档。视频、原始轨迹、日志、下载资产、环境变量/凭据保留在仓库外。所有运行示例均不是安全认证。
