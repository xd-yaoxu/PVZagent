# PVZagent — 只读实时建议助手 v1.2.0

面向原版《植物大战僵尸》（Plants vs. Zombies，重点适配 3-7 泳池十卡槽场景）的 **纯视觉、严格只读** 实时战术伴随侧栏。

> **核心安全原则**：0 游戏键鼠输入 · 0 游戏内存读写 · 0 截图像素落盘 · API 密钥仅本机 Windows DPAPI 加密保存（随时可一键切回纯本地 0 外网模式）。

---

## 核心特性

- **原生 WGC 纯内存采集**：基于 Windows Graphics Capture（C++ 静态 `/MT` 原生 DLL + 独立 MTA 线程）直接从显卡缓冲读取窗口客户区，不读取游戏内存，使用后立即 `fill(0)` 清零内存，磁盘零图像残留。
- **单帧 `< 0.8s` 快速视觉感知 + 僵尸物种/护甲/姿态细粒度识别**：
  - 缓存仿射 + 7 块非数字 HUD/棋盘 NCC 锚点校验配合单遍冻结感知，输出【卡片】/【威胁】/【位置】三通道真实证据，看不清即如实标为未知；
  - 新增纯内存 [`zombie_intel.py`](src/pvz_agent/readonly_advisor_v1/zombie_intel.py)：本地细粒度识别 3-7 泳池 6 大核心物种（普通 / 旗帜 / 路障受损度 / 铁桶受损度 / 冰车 / 潜水三态：`跃入下潜` · `潜伏水中` · `露头啃咬`），并内置 13 种经典僵尸图鉴供看图模式核查；针对潜水僵尸提供缠绕海草、睡莲诱出、睡莲坚果拦截配射手、露头集火与樱桃炸弹等情境化多方案对策。
- **每路现有植物识别与火力充分性评估（防冗余增防）**：
  - 新增纯内存 [`plant_intel.py`](src/pvz_agent/readonly_advisor_v1/plant_intel.py) 与 [`lane_defense.py`](src/pvz_agent/readonly_advisor_v1/lane_defense.py)：逐格识别十卡体系植物（向日葵 / 豌豆 / 双发 / 坚果磨损 / 地刺 / 土豆雷是否出土 / 缠绕海草 / 窝瓜 / 睡莲），并针对低矮陷阱做零误报补扫（P95 `< 3.5 ms`）；
  - 按 PVZ1 物理机制逐路定量结算有效射程 DPS、一次性陷阱消耗、潜水露头窗口与坚果缓冲时间，输出 **`稳守 / 吃紧 / 缺口 / 未知`** 判定：已稳守路立即退出并拦截冗余增防建议，自动转入收阳、补向日葵/睡莲发育或留阳应对下波；
  - 严格执行 **3-7 泳池水陆地形铁律**（1/2/5/6 陆地禁海草与睡莲；3/4 水路禁土豆雷与地刺，陆生植物须先垫睡莲或樱桃相邻 2/5 陆路溅射），并在规则与 LLM 侧双重拦截非法建议。
- **防跳动单侧栏与时效保护**：
  - 独立 2.0s 操作系统看门狗线程：切窗、游戏暂停或选卡界面立即撤下当前建议并释放 WGC，回到游戏自动重连；
  - 固定高度卡片布局（建议历史 **0 px 跳动**）+ 同一侧栏内置「建议」「时间轴（四阶段动态目标）」「设置」三页，支持一键折叠为窄条。
- **极速异步 LLM 战术教练（默认开启，可随时切回纯本地 0 外网）**：
  - **全链路低延迟优化**：支持 `none(关闭思考)` / `minimal(极低)` / `low(极速)` / `medium(标准)` / `auto` 思考档与 HTTP 400 自动逐级回退记忆；静态 `instructions`/`system` 拆分以命中服务端 Prompt Cache + `responses` 默认 `"store": false`；精简 3 字段 JSON 输出（本地回填 `source_state_id` 后走严格策略校验）；支持可选 SSE 流式接收（自动回退非流式）；
  - **事件驱动提前触发与连接预热**：冷却下限支持至 `3s`，新确认冰车/越警戒线/关键缺口路突变满 `3.5s` 防抖即可提前派发；「开始监测」时后台异步执行 0 Token 费用 TCP+TLS 连接池预热；
  - **三档请求模式**：`纯文本极速(不传图)`（默认）/ `轻量看图(512px·low)` / `高清看图(原画·high)`；开启 LLM 模式后仅显示 LLM 建议与历史，局势变动时透明标注 **`（可能过时）`**；
  - **安全记忆**：设置页支持一键切换 `responses` / `chat_completions`、自主填写 OpenAI 兼容 `Base URL` / `API Key`、自动拉取 `GET /models` 与毫秒级连通测试，配置通过 Windows DPAPI（`CryptProtectData`）加密保存在仓库外的 `%LOCALAPPDATA%\PVZagent\readonly_advisor_v1_settings.json`。

---

## 运行环境

- **操作系统**：Windows 10 (1903+) / Windows 11（64 位）
- **Python**：Python 3.11+（含自带 `tkinter`）
- **目标游戏**：原版窗口化《植物大战僵尸》（支持 `800×600` 或 `1200×900` 客户区，进程名匹配 `PVZ-Windowed.exe` / `PlantsVsZombies.exe` / `popcapgame1.exe`；也可通过环境变量 `PVZ_GAME_EXE` 精确指定任意路径）

---

## 快速上手

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 运行离线自检（验证冻结基线与原生 DLL）

```powershell
python scripts/pvz_readonly_advisor.py --self-test
```

预期返回 `"status": "SELF_TEST_PASS"`（`input_sent: false`、`stop_latched: true`）。

### 3. 启动只读实时建议侧栏

双击根目录的 **`启动只读建议助手.bat`**，或运行：

```powershell
pythonw scripts/pvz_readonly_advisor.py
```

1. 点击侧栏顶部 **「选窗(5秒)」**，在倒计时内切换到原版 PVZ 游戏窗口；
2. 如使用 LLM 战术建议（默认已开启），切到 **「设置」** 填写 `Base URL` 与 `API Key`（自动拉取模型列表，并支持本机 DPAPI 加密记忆），按需选择「请求模式（推荐默认纯文本极速）」与「推理强度（推荐 `low`）」；
3. 点击 **「开始监测」** 后切回游戏即可开始实时建议与时间轴追踪（随时可在设置中取消勾选 LLM 开关切回纯本地 0 外网模式）。

> 可选：如需自定义游戏主程序精确路径，启动前设置环境变量 `PVZ_GAME_EXE=你的完整路径\PlantsVsZombies.exe`。

---

## 逻辑架构

```text
原版 PVZ 窗口 (MainWindow, 800x600 / 1200x900)
   │  [L1] OriginalPvzWindowSelector + pvz_wgc_continuous.dll (WGC 内存直采, 边缘 Alpha 门禁)
   ▼
800x600 标准 RGB 内存帧（用完即 fill(0) 清零）
   │  [L2] FastGeometryNormalizer (缓存仿射 + 7 块 NCC) → FastSinglePassPerception (单遍冻结感知)
   ▼
结构化观测状态（阳光 / 10 卡槽 / 僵尸物种与护甲 / 车辆 / 54 格植物与火力充分性 / 地面阳光）
   ├──► [L3 主通道] 本地策略（稳守拦截 + 水陆地形护栏）+ 三通道真实证据 + PhaseTracker (四阶段时间轴)
   └──► [L3 默认开启] AsyncLlmCoach (Prompt Cache 拆分 + 事件驱动提前触发 + 地形与稳守双复核)
   │  [L4] 2.0s OS 看门狗 + 3.5s 停留稳定器 + 切窗/暂停自动撤回与重连
   ▼
[L5] ReadonlyAdvisorSidebarApp (单右侧可折叠固定高度侧栏 + Windows DPAPI 加密配置)
```

---

## 从源码重新编译原生 WGC DLL（可选）

仓库内已附带 64 位 `src/pvz_agent/readonly_advisor_v1/native/build/pvz_wgc_continuous.dll`。如需自行验证并从 C++ 源码重编，在 **x64 Native Tools Command Prompt for VS 2022+** 中运行：

```bat
cl /nologo /c /utf-8 /std:c++20 /EHsc /O2 /W4 /MT /DUNICODE /D_UNICODE /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 ^
  /I src\pvz_agent\readonly_advisor_v1\native ^
  src\pvz_agent\readonly_advisor_v1\native\pvz_wgc_continuous.cpp ^
  /Fo:src\pvz_agent\readonly_advisor_v1\native\build\pvz_wgc_continuous.obj
link /NOLOGO /DLL /MACHINE:X64 /INCREMENTAL:NO /MANIFEST:NO /DYNAMICBASE /HIGHENTROPYVA /NXCOMPAT ^
  /OUT:src\pvz_agent\readonly_advisor_v1\native\build\pvz_wgc_continuous.dll ^
  src\pvz_agent\readonly_advisor_v1\native\build\pvz_wgc_continuous.obj ^
  user32.lib d3d11.lib dxgi.lib dwmapi.lib runtimeobject.lib oleaut32.lib ole32.lib
```

---

## 运行测试

```powershell
python -m unittest tests/test_readonly_advisor_v1.py -v
```

---

## 开源打包导出

如在包含大量历史实验数据的开发工作区中准备开源发布，可运行：

```powershell
python scripts/export_opensource_release.py
```

脚本会在 `deliverables/PVZagent-opensource-v1.2.0/` 生成一份仅包含核心代码、运行必需冻结基线与文档的干净发布目录（约 ~20 MB，不含历史录屏、快捷方式、第三方二进制与私人路径）。

---

## 免责声明与许可

- 本项目仅供计算机视觉、实时人机协作与工程安全边界研究学习使用，**不包含任何游戏本体或第三方商业素材**。
- 本版本严格只读，不读取或修改游戏进程内存，不向游戏发送任何模拟输入。
- 如需配合窗口缩放工具（如 Magpie），请从对应开源项目官方页面自行下载。
- 本项目基于 [MIT License](LICENSE) 开源。
