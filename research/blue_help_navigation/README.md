# Helper Navigation / 向游戏人物问路

**Version 0.4.1 — existing game-character helper, 2026-10-25.**

默认 `trial` / `survey` 使用现有 MPFB GAME v7 neutral，蓝人只保留为显式 `--helper-profile blue-reference` 回归参照。没有新生成人物；GLB 不塞进源码，必须提供已核对的私有本地文件。详见 [游戏角色接入](docs/GAME_HELPER.md)。

这是独立的 Habitat / R2RCE 研究模块，不是 Go2 真机控制代码，也不会被父仓库的 ROS / Go2 启动流程自动执行。源自本项目先前的 v1 交互控制、v2 人工审阅完整路线、v3 单条视觉模型测试准备；旧版代码和实验原件保持不变。**不是恢复被拒绝访问的旧 Library v1.1，也不是完整移植原 hsgmv3 VLM 系统。**

## 已实现 / 尚未完成

游戏角色已实际通过局部接近/阻穿/可见/offer 微检：0.85 m入圈，前进0.25 m到0.60 m，再向人体前进被拦下；转背不可问。合成有效回答交付1次、重复不计，API0。该检查不是全场景拓扑或模型导航成功证明。

- 真实 Habitat-Sim 0.3.3 CPU / Mesa 场景渲染、游戏角色及蓝人参照载入、NavMesh 与人体扫掠检查。
- **1 m** 地面水平圆盘，按假人**脚下中心**判定，边界包括；必须当前可见、同层、无遮挡、短路可达。圈仅用户调试视图显示，不覆盖模型 RGB，不是碰撞体。
- `offer → accept / decline / ignore`；只有有效回答成功交付才原子扣一次，每条任务最多 **3 次**。拒绝、忽略、失败不扣问路次数；模型 API 计费独立。冷却、离圈滞回、超时、重置、重复回调均有测试。
- 人工完整路线：固定的**假人旁安全锚点 → 人工标注终点**。问路时只补车当前位置到这个锚点的短连接及朝向修正。路线全文不再受旧 3 段 / 8 m 限制。
- 英文 policy / question / dummy-oracle 三份手写 prompt；COCO80 实例证据、距离、转弯、镜面反射、终点关系及不确定性要求。
- 单条 episode 40 的真实 CPU 离线调查（蓝人原版与 GAME 替换后分别实跑）已走 **18.9387 m、192 次前进、235 个动作、47 帧**；它使用特权路径，只是路线调查，**不是 VLM 自主导航成绩**。
- YOLO26n 官方权重已在隔离 CPU 环境核对 80 类并测六张历史帧；存在重复检测、chair/couch 冲突和漏检。**没有接入当前实时 policy，也没有宣称每个地标都被检测到。**
- 全长回答的真实 Habitat + 假模型响应接线测试曾通过：交付 1 次、移动 0 m。**真实 VLM 导航尚未执行，成功率 / SPL 未报告。**
- OpenCode Go 接口代码已备好，但账号 **Extra Usage OFF 尚未核实**；本次整理不发送任何 API 请求。模型看图能力仍待一次真实图像 probe 与人工核对。

## 目录

```text
blue_help/
  help_offer.py, interaction_range.py   # 交互状态机与范围规则
  collision.py, habitat_adapter.py      # 真实 Habitat / 人体阻穿与观测
  lost.py, policy.py, runner.py         # 无目标泄漏的决策边界与执行
  manual_route/                        # 完整调查、不可变证据卡、短连接
  trial/                               # 单条试验、视觉 policy、图像 probe
  providers/                           # 严格协议 + 显式授权的有界 HTTP
  evaluation.py, harness.py             # 同 episodes/seeds 的配对评估框架
  prompt_data/                         # 唯一运行时英文 prompt 源
  assets/blue_dummy/                    # 显式回归参照；默认GAME资产为外部私有输入
configs/                               # 交互参数和未执行的配对计划
examples/                              # 明确标注的合成单元测试示例
scripts/                               # 离线校验、CPU 环境设置
tests/                                 # 默认全部离线；无 API / GPU
third_party/                           # 原始 MIT / Apache-2.0 文本
docs/                                  # 数据契约、状态、恢复、来源
```

## 最快验证（不下载数据、不调用模型）

在本目录运行，Python 3.9+（已测 3.9.23）：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu-tests.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m blue_help --help
python -m blue_help plan --batch-size 2
python scripts/verify_source.py
```

`plan` 仅生成计划。默认 2 条任务一批，每条有 no-ask / max3-ask 两个串行条件；最多 4 条任务一批，并发 1。当前只批准准备/试跑一条，**不会因生成计划自动启动 20 条**。两供应商四组仍未安排，没有猜测第二供应商地址或共享其额度。

## 接上已有私有输入（源代码包不附带数据）

详见 [数据契约](docs/DATA_CONTRACT.md)。通过命令行显式提供路径，没有本工作区的绝对路径依赖，没有自动下载：

```bash
python -m blue_help validate-data --data-root /private/blue_dummy_scene_data
# 已安装独立 Habitat CPU 环境后，只测试实际接线，无模型 API：
python -m blue_help trial --fixture-preflight \
  --data-root /private/blue_dummy_scene_data \
  --review-root /private/blue_help_manual_route_v2 \
  --staging-report /private/smoke_demo_relocation_native_r010.json \
  --helper-asset /private/game_character_v7/adult_neutral.glb \
  --output /private/results/new_fixture_attempt
```

`--review-root` 为之前已冻结、包含 47 帧及 manifest 的私有人工审阅包。严格核对 hash、原始英文全文、图像与内容派生 ID。**不执行该输入包中的 Python**；只使用本源码内的已核对契约实现。缺文件/哈希变化直接报错，不静默降级。

## 真模型测试（尚未执行）

先看 [安全与信息边界](docs/SAFETY_AND_PRIVILEGE.md)。仅在已明确授权、Go Extra Usage 关闭且账号相应模型可用时：

```bash
python -m blue_help probe --confirmed-extra-usage-off \
  --image /private/results/new_fixture_attempt/raw_000.png \
  --output /private/results/new_probe.json
# 人工比较 probe 描述与源图后，才允许：
python -m blue_help trial --confirmed-extra-usage-off \
  --confirmed-probe-image-match --successful-probe /private/results/new_probe.json \
  --data-root /private/blue_dummy_scene_data \
  --review-root /private/blue_help_manual_route_v2 \
  --staging-report /private/smoke_demo_relocation_native_r010.json \
  --helper-asset /private/game_character_v7/adult_neutral.glb \
  --max-steps 160 --output /private/results/new_live_attempt
```

密钥只能从交互终端无回显输入临时读取，不从参数、环境变量、文件或聊天自动提取；不保存凭据。上述两个确认参数是**操作者已完成检查的声明，不是代码能够验证账号开关**。不伪装 coding traffic，不绕限流；固定官方域名、无重定向、无自动重试/供应商回退/充值。输出只证明实际执行到的阶段。

## 恢复与发布边界

- [环境与恢复](docs/ENVIRONMENT_AND_RESTORE.md)：环境 recipe / exact lock 可重建，不复制二进制环境到另一台服务器。
- [验证状态](docs/STATUS.md)：区分单元测试、实际仿真、特权调查与尚未完成的模型实验。
- [来源与许可](NOTICE.md)：原仓库固定 commits / assets 哈希；无 DINO / SAM3 或旧验证模块。
- 原始 MP3D、episode 坐标、私有 route card、截图、日志、权重、凭据、环境均不进此 Git 源码目录。`.gitignore` 是防误提交措施，不替代发布前扫描。
- 没有 Go2 动作桥、ROS topic、实机驱动或自动启动关联；未来接机械狗要独立安全适配和验证。
