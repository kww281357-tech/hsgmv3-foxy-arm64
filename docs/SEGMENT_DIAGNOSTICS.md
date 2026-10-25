# 易错片段诊断

`scripts/prepare_failure_segments.py` 从已有结果的执行记录取动作前位置和朝向，生成独立的局部诊断数据集。它不调用模型，不启动仿真，也不自动拆分指令。

cases 文件是一个 JSON 数组，每项指定 `segment_id`、原始官方 `episode_id`、`source_result`、动作前的 `step` 和原指令的连续子串 `instruction`。相对结果路径以 cases 文件所在目录为基准。来源结果须保存同一条完整指令，且包含同层的 `pose_before`、`yaw_before` 和字符串 `executed_action`；不会用去重后的轨迹索引猜测步骤。

```sh
python scripts/prepare_failure_segments.py --dataset DATASET.json --cases CASES.json --output SEGMENTS.json
# 在 src 目录运行；路径相对于该目录。
python run_experiments.py --task r2r --dataset_path SEGMENTS.json --max_steps 24 --no-subtask-decomposition --output_root LOCAL_RESULTS
```

片段元数据强制关闭子任务拆分，优先于配置和 `HSGM_DECOMPOSE_INSTRUCTION=1`。无步数上限或超过 24 步的片段运行会被拒绝。正常数据集的行为保持原样。

这是从舍入后的位姿重新建立地图和历史的局部测试，不是恢复完整检查点。原结果和原数据集不会被覆盖；新文件保留来源哈希。若为旧记录补充推导朝向，应使用独立诊断副本，并在 case 的 `source_adaptation` 中注明方法、原文件哈希和误差。

结果保存 `diagnostic_segment`、实际拆分状态和 `diagnostic_contract_satisfied`。初始化失败时，未知拆分状态写为 `null`，不会冒充关闭。完整 episode 的比较脚本拒绝计入片段结果。保留的目标坐标仅用于原终点的距离评估；中途片段须按局部预期动作判断，不能用原终点距离当作片段成功率。

局部验证应检查实际 STOP、目标身份与几何、阻塞动作、确认转向后的重新验证，以及回溯实际返回位置。仅触发某个分支不等于修复了导航失败。

