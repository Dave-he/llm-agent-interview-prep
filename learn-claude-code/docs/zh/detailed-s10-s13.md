# s10–s13 深度详解：任务系统 · 后台任务 · 定时调度 · 多智能体团队

> 本组覆盖 harness 从"单会话执行"走向"持久化运行时"的四步：s10 给目标一个活过上下文压缩的容器，s11 让慢操作不阻塞主循环，s12 让 Agent 主动到点开工，s13 让多个 Agent 分头共享一个任务板。以顶层 `sXX/` 目录的 17-stage 主线代码为准；`docs/zh` 旧版 12-stage 文档编号与代码不一致（s07-task-system↔s10、s08-background-tasks↔s11、s09-agent-teams↔s13，cron 无旧文档），仅作术语参考。
> 研究口径：《个人 AI Agent 系统全面调研报告（v1.1）》——"模型是地板，harness 是天花板"（考点 19）、LangGraph checkpoint 与人审卡点（表六/考点 10、18）、多智能体"编排确定性 + 模型智能"共识与失败级联（表六/考点 9）。

## 本组导航

| Stage | 主题 | 解决的 harness 问题 | 对应研究文档原理 |
| --- | --- | --- | --- |
| [s10_task_system](#s10-task-system依赖图与状态外置) | 任务系统 | 目标与进度活在上下文之外，可恢复、可排依赖 | 考点 7 短期/长期记忆分野；考点 17 任务分解；考点 10 缩小任务粒度 |
| [s11_background_tasks](#s11-background-tasks异步执行与通知注入) | 后台任务 | 慢命令不阻塞 agent loop，结果后续投递 | 考点 6 上下文工程；考点 2 agent loop；考点 14 成本 |
| [s12_cron_scheduler](#s12-cron-scheduler到点投递-prompt-的确定性外壳) | 定时调度 | harness 记住"何时开工"，替代用户复读 prompt | 考点 1 编排确定性 vs 模型智能；考点 18 HITL；考点 10 检查点 |
| [s13_agent_teams](#s13-agent-teams拓扑协议与冲突解决) | 多智能体团队 | Lead 分工 + 持久队友 + 收件箱 + worktree | 考点 9 多智能体取舍；考点 13 沙箱边界；表六"编排确定性+模型智能" |

---

## s10 Task System：依赖图与状态外置

### 1.【定位与动机】

s05 的 TodoWrite 是会话内清单：每项只有内容与状态，harness 无法回答"这个任务现在能不能开始、由谁负责"。一旦大目标拆成"建 schema → 写 API → 写测试"，harness 必须知道先后关系；一旦引入多 agent（s13），还必须知道归属。研究文档考点 7 的记忆分野正好落在这一层：短期记忆（上下文窗口）会被 compact 吃掉，长期记忆必须外置——README 的"关键洞察"一句话点破：*让状态活过压缩，因为它在对话之外*。这也是考点 10 的工程对策之一："把任务拆小 + 检查点持久化"，任务 JSON 文件就是最小粒度的检查点。

### 2.【源码精读】

核心数据结构是一个极简 dataclass（`s10_task_system/code.py:74-81`）：

```python
@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None
    blockedBy: list[str]
```

`TaskStore`（`s10_task_system/code.py:84-199`）负责落盘：ID 由 `secrets.token_hex(4)` 随机分配、正则 `task_[0-9a-f]{8}` 语法校验 + `is_relative_to` 路径封锁（`code.py:96-103`），建文件用排他写 `"x"` 防撞（108-131），load 时校验文件内 id 与文件名一致、status 在白名单内（185-192）——防的是模型幻觉出的 ID 和被注入 path traversal 的任务 ID。

两个最重要的算法：环检测（`code.py:133-145`）用 DFS 沿 `blockedBy` 递归判断 `task_id` 是否传递依赖 `target_id`，在 `update_dependencies`（147-177）里拒绝成环：

```python
if dependency not in task.blockedBy and self._depends_on(
    dependency, task_id
):
    raise ValueError(
        f"Dependency cycle detected: {task_id} -> {dependency}"
    )
```

以及 `complete_task` 的解锁通知（`code.py:254-279`）：先把"改动前本就 ready 的任务"算进 `ready_before` 集合（260-266），完成后再扫一遍全表取差集，只把**本次完成真正解锁**的下游报给模型——避免模型每完成一个任务都被重复噪声轰炸。主控制流上，`claim_task`（240-251）校验 pending + 依赖全完成后才设 owner 转 `in_progress`；系统提示词写死了两阶段契约（`code.py:60-65`）："先 create 所有节点，create_task 返回运行时 ID 后，再用这些 ID 调 update_task 建依赖边"。

### 3.【设计权衡】

**为什么两阶段建边？** agent loop 里同一响应的多个 tool_use 在结果返回前就已确定参数，`create_task` 拿不到同批调用刚生成的 ID。硬逼模型猜 ID 只会产出错误边；把"建边"拆成显式第二阶段，并用 schema 的 `pattern` 字段（`code.py:409`）逼模型必须用运行时返回的真实 ID。

**为什么 ID 随机而不是自增？** `agents/s07_task_system.py` 是前身：TaskManager 用 8 字节自增 int + `_clear_dependency`（其 `code.py:101-107`，完成时把 id 从其他任务列表里删掉）+ 无环检测、无 claim/complete。s10 升级为随机八位 hex：自增 ID 与"创建顺序"耦合，多 agent 并发创建时排序碰撞是坑；不改写他人列表（blockedBy 永久保留、靠 `can_start` 时算）则让完成操作幂等、也保留了历史依赖的可审计性。

**对照真实 Claude Code：** 真实 CC 的 TodoWrite 仍是会话内、精简状态机的清单（内容+两种状态），分布式任务跟踪、owner/依赖这类"任务数据库"能力在真实 CC 里属于上层运行时（Task/团队运行时的方向）。教学版提前把这一层做厚，正是为了讲清"状态外置"与"多 agent 协调地基"这两件事。若裁撤环检测：图变成脏数据，complete 会静默永远等不到依赖；若换成"进程内存任务表"（裁掉落盘）：断点恢复能力消失，s13 的跨线程共享任务板也无从谈起（共享状态必须先有"共享存储"）。

### 4.【面试考点映射】

主考：**考点 17（agent 的规划怎么做）**、**考点 7（记忆系统设计）**，兼答考点 10 检查点。
30 秒框架："规划在工程上就是 plan-first 的任务分解 + 执行中重规划；分解结果不能活在上下文里——compact 之后就丢了，所以要外置成带依赖图的任务状态库。每个任务有 id/status/owner/blockedBy 三态生命周期，claim 前做依赖检查，complete 时广播解锁。这等价于把 LangGraph 的图状态降维成文件级检查点，换来跨会话与跨 agent 的可恢复性。"
追问 1：*TodoWrite 和 Task System 边界在哪？*——TodoWrite 面向"我自己这轮怎么干"（会话态、行粒度）；Task 面向目标与协作（持久、可认领、有依赖）。
追问 2：*依赖检查为什么放 claim 而不是调度器？*——保持"harness 只挡、模型选"的 agent loop 本色，claim 是模型的决策点，检查只保证这个决策不可违反不变式。

### 5.【动手实验】

打开 `s10_task_system/code.py:240-246`，把 `claim_task` 里的依赖检查短路：

```python
# dependencies = incomplete_dependencies(task)
# if dependencies:
#     return f"Blocked by: {dependencies}"
```

然后问模型："create tasks: schema → endpoints → tests（按依赖串接），然后直接 claim tests 并完成它"。观察：任务直接进 `in_progress`，模型基于"tests 已完成"的假象继续生产代码。这验证的设计假设是——agent 的自主规划必须靠 harness 的不变式兜底，否则模型会把"想 claim"当成"可以 claim"，正是研究文档链式衰减难题的最小复现版。

---

## s11 Background Tasks：异步执行与通知注入

### 1.【定位与动机】

s10 的任务图解决了"做什么"，但同步执行的 agent loop（s10_task_system/code.py 里的 `subprocess.run(..., timeout=120)`）在装依赖、跑全量测试时会整段停在一条 Bash 上，停住的不只是当前工具调用，还有整个上下文推进。s11 把"开始执行"与"拿到结果"拆开：`run_in_background=true` 的命令在后台线程跑，主循环立刻拿到 `bg_id` 占位结果继续干活，完成结果在后续轮次以 `<task_notification>` 形式回灌。这正是研究文档考点 2 里"观察反馈再决策"循环的异步化：观察可以延迟到下一轮，但循环不能死等。README 的点题句：*慢操作放后台，Agent Loop 继续运行*。

### 2.【源码精读】

入口判断极窄（`s11_background_tasks/code.py:432-436`）：只有 bash 且 `run_in_background is True` 才走后台——不按关键词猜，是否异步由模型经 schema 显式声明（工具 schema 见 202-229 行 bash 的 `run_in_background` 参数）。

`BackgroundManager`（`code.py:348-424`）是三态生命周期管理器。`start()`（356-384）在锁内登记 `bg_NNNN`、保存原始 `tool_use_id`，起 daemon 线程立即返回 ID：

```python
with self._lock:
    self._counter += 1
    task_id = f"bg_{self._counter:04d}"
    self.tasks[task_id] = {
        "tool_use_id": block.id,
        "command": command,
        "status": "running",
    }
```

worker `_run()`（386-401）执行后写 status（completed/failed）和结果，并把自己放进 `_ready` 完成队列。投递侧 `inject_background_results()`（447-467）在**每次 LLM 调用前**被 `agent_loop` 调（493-503 的 `inject_background_results(messages)`），把通知块合并进**最后一条 user 消息**、避免产生相邻 user 消息（合并遥不可及分支的注释写明这是为独立复用保留的主循环不变式）。通知格式（414-422）带自闭合标签与 500 字摘要：

```python
f"<task_notification>\n"
f"  <task_id>{task_id}</task_id>\n"
f"  <status>{task['status']}</status>\n"
f"  <command>{command}</command>\n"
f"  <summary>{result[:500]}</summary>\n"
```

进程生命线是本章隐藏主角：`code.py:60-107` 维护 `_shell_processes` 全集，POSIX 用 `start_new_session=True`（113-114）让命令独立成进程组，守卫线程 `os.killpg` 先 SIGTERM 后 SIGKILL 逐级清理（64-91），并注册 `atexit` + SIGTERM handler（101-107）——进程退出时后台命令不留孤儿。

### 3.【设计权衡】

**为什么通知不复用原 `tool_use_id`？** 原始 tool call 已答复过占位 tool_result；Claude API 的不变式是"一个 tool_use 对应恰好一个 tool_result"，二次回填同 ID 直接触发协议错误。所以放弃协议伪装，用运行时事件格式 `<task_notification>` 走 user 通道——这保持了协议正确性，代价是模型需要理解一种新事件类型。（`agents/s08_background_tasks.py` 的前身用 `background_run`/`check_background` 两个独立工具 + 通知队列在 LLM 调用前排空，`s08:210-229`；主线版收敛为单工具参数 + 自动注入，交互面更省 token。）

**对照真实 Claude Code：** 真实 CC 的 Bash 工具同样有 `run_in_background`，后续用 `TaskOutput`/`KillShell` 类工具轮询与收割，后台任务死亡时推送 `<task-notification>` 事件——教学版把"轮询收割"简化成"每轮循环前自动注入"，等于把工具轮询的决策也收归 harness。若裁掉进程组清理（删 60-107 段）：后台 npm install 会在会话退出后变孤儿进程锯肉；若把通知合并分支改成强制新建 user 消息：API 立即报相邻 user 消息错误——远处看这就是"agent 框架必须处理的消息序列化问题"的教具。这也呼应"harness 是天花板"（考点 19）：模型只会声明"我想异步"，能不能安全异步、结果何时进上下文、开销多大（每轮都注入、摘要只给 500 字防 context rot），全部由 harness 层决定。

### 4.【面试考点映射】

主考：**考点 6（context engineering）**，兼答考点 14（成本）。
30 秒框架："异步执行的核心是结果如何再进上下文。保存原始 tool_use_id 会导致一个 tool_use 两个 tool_result，违反 API 不变式，所以后台完成以带自闭合标签的 task_notification 事件注入，并在每轮 LLM 调用前合进尾部 user 消息——既不产生相邻 user 消息，也天然击中 prompt 缓存前缀；摘要截断 500 字是在控制 context rot。"
追问 1：*为什么不在线程完成时直接中断主循环注入？*——主循环正持有 LLM 流式响应或正在写 messages，重入会撕开会话一致性；对面 myflicker/harness 的做法也是 defer 到进入下一轮 loop 前（`inject_background_results` 的可达性注释就是证据）。
追问 2：*后台任务无超时上限会怎样？*——事实上 worker 里 `communicate(timeout=120)`（code.py:126）与前台一致，若放开上限必须补 KillShell 语义，否则失败命令占住结果队列并污染后续轮次。

### 5.【动手实验】

验证不变式实验：把 `s11_background_tasks/code.py:452` 的通知块换成协议伪装：

```python
blocks = [{"type": "tool_result",
           "tool_use_id": task["tool_use_id"],
           "content": item} for item in notifications]
```

注意 `start()` 已把 `block.id` 存进 `self.tasks[task_id]["tool_use_id"]`（370 行）。跑一句"run pip list in the background then read package.json"，第二次 API 调用会直接报 tool_use_id 重复对应的 400 错误——用一行改动证明"一个 tool_use 只有一个 tool_result"是框架必须主动维护的不变式，也解释了为什么通知只能走独立事件通道。

---

## s12 Cron Scheduler：到点投递 prompt 的确定性外壳

### 1.【定位与动机】

s11 使命令"开始之后"不阻塞，但没有任何组件回答"这件事应该几点开始"。"每天 9 点跑测试"的痛点是：用户每轮到点都要复读 prompt。s12 给 harness 装一个闹钟：cron 表达式 + prompt 注册成 job，调度线程每秒对表，到点先进队，队列处理线程等 Agent 空闲后投递一轮新的 agent loop。架构站位与研究文档考点 1 完美咬合：cron 表达式、队列、 Locke 层面的重复去重全是**编排确定性**（workflow 属性），而 prompt 交给 Agent 后发生什么仍完全由**模型智能**驱动——README 与研究文档表六"多智能体框架以编排确定性+模型智能为设计共识"是同一句话在这层的投影。

### 2.【源码精读】

CronJob（`s12_cron_scheduler/code.py:269-277`）带两个交付级状态位：`pending_delivery`（已到点、模型没收到）和 `last_fired`（分钟标记，防同分钟重复入队）。cron 匹配器 `cron_matches`（299-321）支持 `*`、`*/N`、`N`、`N-M`、`N,M` 并实现 Vixie cron 的 day/weekday 限入 OR 规则，Python `weekday() + 1) % 7` 把 Python 周一为 0 转成 cron 周日为 0 的惯例（305）。

调度主循环每秒一次对表（`code.py:644-646`），`poll_due_jobs`（497-508）按分钟标记去重：

```python
minute_marker = moment.strftime("%Y-%m-%d %H:%M")
with cron_lock:
    for job in list(scheduled_jobs.values()):
        if job.pending_delivery or job.last_fired == minute_marker:
            continue
        if cron_matches(job.cron, moment):
            _enqueue_due_job(job, minute_marker)
```

`_enqueue_due_job`（481-494）是"持久化先、入队后"的顺序：先把 `pending_delivery=True` + `last_fired` 写盘（`save_durable_jobs` 用临时文件 + `os.replace` 原子替换，387 行），失败则回滚内存状态——保证"进了内存队列的任务必先活过重启"。

交付侧 `agent_loop`（649-699）开头 `consume_cron_queue()` 拿走到期任务，作为 `[Scheduled] {prompt}` 消息追加（653）；注意讨论 ack 时序：`waiting_for_ack` 只有在**首次成功收到模型响应**后才 `acknowledge_cron_jobs`（673-679，非 recurring 的任务被删除）；若 API 调用异常，`del messages[scheduled_start:]` 抹掉已加消息 + `restore_cron_jobs` 复位回队（666-671）——这就是研究文档考点 10 "检查点与回滚"的最小可运行实现。第三个线程 `queue_processor_loop`（727-735）用 `agent_lock` 非阻塞抢锁投递，与用户交互轮互斥。`request_permission`（194-203）加了线程身份检查：后台定时轮不能申请交互确认，直接拒绝。

### 3.【设计权衡】

**Harness 进程内定时器 vs 系统 crontab？** 代码显式承认局限（README 运行边界）：Agent 进程死了调度也死、durable 只保 job 定义不补跑错过的时刻。选"进程内 cron"换来的是零外部依赖和与 agent loop 同进程的锁模型；研究文档的 LangGraph checkpoint（表六）+ 我方 environment 里的 durable cron 是这一权衡的另一边（跨进程、跨天、补跑语义）。教学版用 `.scheduled_tasks.json` 原子写 + 启动校验重放把 durability 组件教完整即可。

**对照真实 Claude Code：** 真实 CC 本体没有内置 cron——"到点触发一个 agent"在产品里是**上层编排**（CI 定时器、SDK 无头调用 `claude -p`、或像 Factory Droid 的 IM 触发，研究文档表二）。把调度做进 CLI 进程意味着吃满它的生命周期假设。而"定时轮不能做人审"这个细节对应考点 18 的 `interrupt-on-external-effect`：自动触发的 turn 遇到高危命令只能拒绝，不允许背后一个没有上下文的用户盲按 y——这是研究文档"缓冲 + checkpoint 后批量人审"反向操作的正确姿势。

裁掉 `last_fired` 会得到重复投递（同一分钟连发）；裁掉 `pending_delivery` + restore 路径，一次 API 抖动就会永久丢任务（at-least-once 退化为 at-most-once，恰好挂在持久化与交付之间）。

### 4.【面试考点映射】

主考：**考点 1（Agent vs Workflow）与考点 18（HITL）**，兼答考点 10 回滚。
30 秒框架："cron 属于 harness 层的编排确定性：时间匹配、去重、持久化、空闲投机性投递全部是确定性代码；进了 agent loop 才交给模型智能。可靠性用两态投递位 + 分钟标记去重 + ack 后再清理实现至少一次投递——API 失败时把已追加的消息整段删掉、任务复位回队。HITL 的关键规则是：自动触发轮没有真人上下文，只允许拒绝危险命令，不能替用户盲批。"
追问 1：*at-least-once 会重复，怎么幂等？*——`last_fired` 每分钟一个版本号，进度真实现"进过上下文即审过"，重复投递被任务 ID 与我们的消息 schema 前缀 `[Scheduled]` 双重识别，天然幂等；换成"任务卡片"时追 checksum。
追问 2：*为什么队列处理线程不叫起 agent loop 去抢用户正在用的锁？*——`agent_lock` 非阻塞获取（729-735）保证定时轮只在用户轮结束后插入，避免两条 LLM 调用写一份 messages 的数据竞争。

### 5.【动手实验】

打开 `s12_cron_scheduler/code.py:498`，把分钟精度标记改成秒级：

```python
minute_marker = moment.strftime("%Y-%m-%d %H:%M:%S")
```

然后让 Agent：`Schedule "run date" every 1 minute recurring`，观察日志：到点进入工作时 `last_fired` 已含秒，下一秒 marker 变化 → Acknowledged 清 `pending_delivery` 后下一轮立刻重新匹配 → `[Scheduled]` 消息在 1 分钟内反复投递模型十几次直到下一个完整分钟到来。这个实验可证伪 `last_fired` 的一个精妙声明：**去重标记的状态位必须和触发对齐粒度一致**——评估是日历预测、审计、还是对账，粒度选择直接影响是否重复触发、耗不耗 token。

---

## s13 Agent Teams：拓扑、协议与冲突解决

### 1.【定位与动机】

s06 的 subagent 是一次调用即焚的隔离系统，s13 把它升级为**持久队友**：Lead 对接用户、拆分任务、经用户确认后启动具有身份标识的 Agent 线程，队友在 WORK 与 IDLE 之间循环直到关机协议结束。动机链正好对应研究文档考点 9"什么时候用多 agent"的三个主条件之一：上下文隔离（队友的消息数组独立，工作痕迹不污染主对话）+ 并行探索 + 异构分工（role 字段）。同时这章回答了研究文档表六的多智能体可靠性难题：**通信靠文件、协议靠 request_id、冲突靠闭锁、目录靠 worktree**，全部确定性可推断，不让模型自由发挥。

### 2.【源码精读】

**任务存储的跨层锁**（`s13_agent_teams/code.py:70-89`）：`task_store_lock` 用 `threading.RLock` + `threading.local` 层次计数 + `fcntl.flock` 文件锁三层，进程内重入、进程间互斥——同一任务目录可被多个 harness 进程共享（开头 docstring 明确说 `.worktrees/` 是可选项，多进程共享是设计目标）。`Task` 增加 `worktree` 字段（112-120），写任务用临时文件 + `os.replace`（205-217）。

**原子认领**（`claim_task`，281-308）：整个检查-修改-落盘序列握在 `task_store_lock` 里，叠加"一个队友一次只许一项任务"的单一 work 契约检查（289-296）：

```python
assignment = teammate_assignments.get(owner)
if assignment:
    return (f"Owner {owner} must finish the current work turn for "
            f"{assignment['task_id']} before claiming another task")
...
task.owner = owner
task.status = "in_progress"
save_task(task)
teammate_assignments[owner] = {"task_id": task.id, "cwd": cwd}
advance_assignment_version(owner)
```

`advance_assignment_version`（92-109）把 work_version 加一，使旧的 plan 审批自动过期（`apply_plan_response` 1060-1088 校验 version 与 task_id 双匹配）。

**信箱总线** `MessageBus`（852-909）：`.mailboxes/<name>.jsonl` 追加写、**破坏性读**（读完即删，867-874），锁 + `Condition` 实现 `wait_for_messages`（898-909）——队友 IDLE 时不空转轮询，事件到达即唤醒。收件箱不是模型工具：消费发生在 `consume_lead_inbox`（970-980，先更新协议状态机再给模型看）和 CLI 主循环 `wait_for_cli_event`（1836-1851，`select` 等一根标准输入管道 + 250ms 突发轮询 `BUS.peek("lead")`）里。

**协议** `ProtocolState`（921-934）记录 type/sender/target/status + `work_version`，`match_response`（944-967）校验类型、方向、状态三层后才改状态。**计划闸门**（plan gate）在工具分发层执行（1039-1044）：`required/pending/rejected` 状态下队友的 `bash/write_file/edit_file` 直接被拒，等 plan 获得 lead 审批过才放行。**工作循环** `work()`（1271-1320）把退出区分成两个事件（1307-1319）：先发 `result`（任务产出），再发 `idle_notification`（我能接活了）；以及 `wait_for_work`（1322-1346）：IDLE 时**先看收件箱**（关机、lead 指令优先），无消息才扫任务板 `claim_next_task`（1139-1148）。整个拓扑：Lead 是唯一能 `spawn_teammate/create_worktree/update_task` 的角色，TEAMMATE_TOOLS（1575-1591）只给队友 list/claim/complete，不给 create/update 任务。

### 3.【设计权衡】

**对照 s06 subagent 与研究文档的多代理取舍：** 完美契合考点 9 的决策框架——多 agent 用于并行探索 + 上下文隔离，代价是成本倍增、失败级联、handoff 信息丢失。s13 对三大代价逐个下药：成本用"先提方案，用户确认"的边界（system prompt 641-655）+ Lead 主流程提示"idle 队友不退出、shutdown 显式"；失败级联要求任务板只传递状态、每个队友独立循环死了不发 `error` 事件（run() 异常上报 BUS，1355-1361）；信息丢失用 result 事件 + 任务板 description 双通道保真。真实 Claude Code 的团队形态（研究文档未展开，2025 底的产品形态）也是 persistent teammates + 通信 + shared tasks。行业共识"编排确定性 + 模型智能"：**拓扑由 Lead 生成、兜底由运行时锁与协议兜**。

**worktree 的杀伐范围：** README 明写"worktree 只是工作目录隔离、不是沙箱"（源码 `bash` 依然 `shell=True` 无资源限制），代码里 `create_worktree` 精确处理 git 部分失败（489-565 partial operation 保留现场），删除（remove_worktree 568-618）**保留 `wt/<name>` 分支**。这正是研究文档考点 13 "代码执行放容器/worktree 并默认禁网"的半成品：目录隔离，网络/资源未隔离——和主流程明确分工，留给后续 stage 或真实沙箱方案。

裁掉 `task_store_lock` 多层结构 → teammate 的认领竞争可能同 claim 同一任务（文件级 write "x" 只排了"创建"竞争，没排 owner 改写竞争）；裁掉 plan gate → `require_plan=True` 是纯提示词"许愿"，模型可以偷偷改文件；裁掉 `advance_assignment_version` → 队友带着 A 任务的审批串到 B 任务执行。

### 4.【面试考点映射】

主考：**考点 9（多智能体取舍）**，兼答考点 10 可靠性、考点 13 沙箱。
30 秒框架："多 agent 的收益是并行与上下文隔离，成本是 token 倍增 + 失败级联 + 交接信息丢失，默认先单 agent + subagent，出现显著瓶颈才升级。s13 的实现给出最小完整答案：通信走文件总线（破坏性读 + Condition 唤醒），结果与空闲拆成两类事件免歧义；冲突用'检查-修改-落盘'一锁全包的原子认领解决；控制流用带 request_id 的类型化协议；高危工作前加 plan gate。编排确定性与模型智能分工明确——Led 提案给用户确认、协议由运行时强制，模型只做'下一步干什么'的语义决策。"
追问 1：*为什么扫描和认领必须两步？*——scan 只是快照，竞争点在 claim；把所有权变更塞进锁原子区，是"乐观发现 + 悲观提交"的标准模式（一和 agents/s09 比对：前身无锁且一 transfémervre read 例如可能多人拿到同一文件）。
追问 2：*worktree 之间会互相冲突吗？*——git 数据层隔离，但输出目标（共享 `.tasks/`、公共文件、测试跑同一端口）仍在共享层冲突，所以完整答案要"目录隔离 + 资源配额 + 共享文档锁"三件套；这正是研究文档"worktree 隔离并行"（Factory Droid 表二）的边界说明。

### 5.【动手实验】

在 `s13_agent_teams/code.py:1039-1044` 注释掉计划闸门：

```python
# gate = plan_gates.get(name, "not_required")
# if block.name in {"bash", "write_file", "edit_file"}:
#     if gate != "approved":
#         if gate != "not_required":
#             return (...)
```

然后跑"先提方案，确认后 spawn 一个 require_plan=True 的队友"。观察：队友第一次响应就扔出 `bash` 调用并真实执行（闸门只剩首条消息里的 `[Plan required]` 提示词，纯提示），Lead `review_plan` 的拒绝/批准再也管不住行为。这验证的设计假设是：**协议要生效必须落点在工具分发层，系统提示词只是沟通，不是约束**。

---

## 面试火力图

```
                         Agent 原理火力覆盖（本组 s10-s13）

 考点 17 规划/分解      ████████░░  s10 两阶段任务图 + claim 闸门（辅 A）
 考点 7  记忆系统       ███████░░░  s10 状态外置、跨会话恢复（辅 B）
 考点 6  上下文工程     ████████░░  s11 通知注入 + 尾部合并 + 摘要裁剪
 考点 2  Agent Loop    ██████░░░░  s10/s11 循环内嵌执行变化
 考点 14 成本优化       ████░░░░░░  s11 摘要/单参数工具省 token（辅 C）
 考点 1  Agent vs WF   ████████░░  s12 编排确定性 + 模型智能切分
 考点 18 HITL           ████████░░  s12 定时轮禁交互审批；s13 team 需用户确认
 考点 10 可靠性         ███████░░░  s12 at-least-once + 回滚；s13 原子认领
 考点 9  多智能体取舍   █████████░  s13 全章主答案
 考点 13 沙箱/权限      ██████░░░░  s13 worktree 边界 + permission 线程规则
 考点 19 harness 天花板 ████████░░  四章合读：工具不变式/协议/锁都在 harness

 (A) 30 秒答"规划怎么做"可串 s05 TodoWrite → s10 Task → LangGraph 图
 (B) 30 秒答"记忆分咋设计"可用 s10 对比 Letta 分层与 mem0 抽取操作
 (C) 成本的完整答案还要 s06 子代理隔读 + prompt 缓存命中（s11 注入合并）
```

一图流使用建议：面试被问到 Top20 中 1/2/6/7/9/10/13/14/17/18/19 任何一题，都能从本组找一个 stage 作为"我亲手读过源码"的落地证据；最锋利的三把火是 s10 的两阶段建边契约、s11 的 tool_use 1:1 tool_result 不变式、s13 的"认领原子性 + 协议落点在工具分发层"。
