# 公开底座与现场前提

本包没有既有实现需要移植，也没有必须恢复的历史分支或未提交 patch。以下都是公开来源，获取后仍要核对实际运行环境。

## 上游

| 项目 | 公共 Git 仓库 | 交接核验的候选正式 tag |
| --- | --- | --- |
| vLLM | https://github.com/vllm-project/vllm.git | v0.29.0 |
| LMCache | https://github.com/LMCache/LMCache.git | v0.5.5 |

核验日期：2026-09-21。开始实施时检查 [vLLM 正式发布入口](https://github.com/vllm-project/vllm/releases/latest) 与 [LMCache 正式发布入口](https://github.com/LMCache/LMCache/releases/latest)，冻结实际 release tag 和完整 SHA。本包没有部署或端到端兼容通过的声明。

候选版本的公开说明：
- [vLLM v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [LMCache v0.5.5](https://github.com/LMCache/LMCache/releases/tag/v0.5.5)
- [LMCache 与 vLLM 兼容性说明](https://docs.lmcache.ai/getting_started/compatibility.html)
- [vLLM CUDA 依赖（固定 tag）](https://github.com/vllm-project/vllm/blob/v0.29.0/requirements/cuda.txt)
- [LMCache 构建配置（固定 tag）](https://github.com/LMCache/LMCache/blob/v0.5.5/pyproject.toml)

以下是在空白开发目录获取上述候选版本的示例；若实际冻结 tag 不同，替换后记录。目录名仅是示例，不是继承的项目组织规定：

```bash
git clone --branch v0.29.0 --depth 1 https://github.com/vllm-project/vllm.git vllm
git clone --branch v0.5.5 --depth 1 https://github.com/LMCache/LMCache.git LMCache
git -C vllm rev-parse HEAD
git -C vllm describe --tags --exact-match
git -C LMCache rev-parse HEAD
git -C LMCache describe --tags --exact-match
```

仅在相应目录不存在时执行 clone。若已有服务器文件，先核对其用途，不覆盖。依赖安装命令由实际 driver、GPU 架构、Python、torch/CUDA 和 release ABI 决定，使用独立环境。记录扩展源码/构建方式和实际导入位置，不能只记录 pip 版本。

## LMCache 集成的路径检查

优先研究 LMCache MP，候选配置如下；具体字段以最终 release 为准：

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both"
}
```

明确加载外部模块，检查实际源文件，不假定同名 vLLM 内置实现等价。

- [MP connector 固定 tag 源码](https://github.com/LMCache/LMCache/blob/v0.5.5/lmcache/integration/vllm/lmcache_mp_connector.py)
- [DAX 使用说明](https://docs.lmcache.ai/mp/l2_storage/dax.html)
- [DAX adapter 固定 tag 源码](https://github.com/LMCache/LMCache/blob/v0.5.5/lmcache/v1/distributed/l2_adapters/dax_l2_adapter.py)

已核查的 v0.5.5 DAX adapter 是 L1→DAX 保存、DAX→L1 恢复；默认 CPU L1 时会经过 DRAM。这是代码起点，不能直接当成本任务的平级直达实现。slot/chunk、布局、buffer 约束和重启后的目录语义都应按最终 tag 核对。

版本升级可能改变 runner、KV layout、CUDA graph 内存占用、batch 默认预算、异步提交和完成处理。需冻结实际参数，核对 prompt 与 output KV 的保存范围、尾部在途操作和 buffer 生命周期，不用旧环境常数代替本机测量。

## 需现场提供或检测的条件

- 可进行无干扰测量的 GPU，以及真实 CXL 设备/内存区域。
- CXL backing 对 GPU 复制是否可用、需要何种映射/注册，以及实际物理通路。
- 可运行目标模型的显存/内存、模型权重/tokenizer；优先使用已下载模型。
- 访问公开源码与所需发行构件的网络，或服务器已有等价离线制品。
- 可在用户工作区构建和运行的权限。系统级变更另行说明和确认。

这些条件尚未在目标服务器核实。无 CXL 时，普通 DRAM 后端和状态机测试可以继续，但不能产出物理 CXL 校准结果。

## 包的边界

本包携带完整需求与公开恢复入口，不携带第三方依赖、模型、环境、凭据或个人配置。测试文本由新代码生成，因此没有外部 workload 下载前置条件。所有本机配置、脚本、输入和运行结果由接收端的新项目产生并维护。
