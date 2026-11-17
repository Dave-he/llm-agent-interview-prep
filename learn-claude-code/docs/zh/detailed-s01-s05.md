# s01–s05 详解：Agent Loop / Tool Use / Permission / Hooks / TodoWrite

> 本文是顶层 `s01_agent_loop` ～ `s05_todo_write`（以仓库根目录代码为准）的逐 stage 深度详解，配合《个人 AI Agent 系统全面调研报告》考点 Top20 使用。
>
> **本组导航**
>
> | Stage | 目录 | 主题 | 精简版对照 | 核心考点 |
> |---|---|---|---|---|
> | s01 | `s01_agent_loop/` | Agent Loop 与终止条件 | `agents/s01_agent_loop.py` | 考点 1/2/3/19 |
> | s02 | `s02_tool_use/` | 工具 schema 与分发 | `agents/s02_tool_use.py` | 考点 4/14/19 |
> | s03 | `s03_permission/` | 分层授权模型 | （无 agents 版） | 考点 12/13/18 |
> | s04 | `s04_hooks/` | 不可绕过的生命周期钩子 | （无 agents 版） | 考点 12/18/19 |
> | s05 | `s05_todo_write/` | 任务状态机与催更 | `agents/s03_todo_write.py`（旧编号） | 考点 2/17/9 |
>
> ⚠️ `docs/zh/sXX-*.md` 是旧版 12-stage 课程，编号与本组不一致（旧版 s03 是 todo_write），只在术语上参考，勿当主线。`agents/` 目录同样采用旧编号：`agents/s03_todo_write.py` 对应的是顶层 `s05_todo_write`，没有 permission/hooks 两个阶段的精简版。

---

## s01 Agent Loop：一个循环 + 一个工具就是 Agent

### 1.【定位与动机】

LLM 只会生成文本与结构化调用请求，不会自己执行任何东西。s01 之前的世界是：你问"帮我看看目录里有什么并跑一下 xxx.py"，模型吐出一条命令就停了，你得自己执行、把输出贴回去，如此往复——**人就是 agent 的 harness**。s01 把"人搬运"这个角色自动化，这就是 agent loop 的全部秘密，也是研究文档考点 2"感知-规划-行动-观察"和考点 1"Agent 与 Chatbot 的本质区别（控制流由 LLM 决定）"的代码化：模型请求工具就继续循环，模型不再请求工具（纯文本回复）就终止——**下一步走不走、往哪走，由模型的输出决定，不写死在代码里**。这正是"Agent = 模型智能，Workflow = 编排确定性"分界线的第一次落地。docs/zh 旧版 s01 的标题"One loop & Bash is all you need"说的就是这 30 行核心循环。

### 2.【源码精读】

主控制流在 `s01_agent_loop/code.py:96-126`：

```python
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't call a tool, we're done
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            return
        # Execute each tool call, collect results
        results = []
        for block in tool_calls:
            output = run_bash(block.input["command"])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        # Feed tool results back, loop continues
        messages.append({"role": "user", "content": results})
```

三个要点：**(a) 终止条件**——把响应里的 `tool_use` 块全部抽出来（`code.py:107-111`），一个都没有就 `return`，即"模型自停"这一类终止；外层 `code.py:135-146` 的 REPL 循环加上 `q/exit` 和 EOF，构成"人工中断"类终止。精简版 `agents/s01_agent_loop.py:90` 用 `response.stop_reason != "tool_use"` 判定，语义相同但更依赖 API 语义层——两个版本恰好演示了"看块内容"和"看 stop_reason"两种等价判定。**(b) 执行体**——`run_bash`（`code.py:80-92`）先做一张可疑命令黑名单（`rm -rf /`、`sudo`…），再 `subprocess.run(shell=True, timeout=120)`，输出合并 stdout/stderr、截断到 50000 字符，空结果补 `"(no output)"`。**(c) 唯一状态**——`messages` 列表就是 agent 的全部记忆，`tool_result` 以 `role=user` 回填（`code.py:126`），这是 Anthropic API 的约定格式：观察结果作为下一轮输入出现在 user 回合里。

### 3.【设计权衡】

最大的取舍是**循环没有步数上限**：终止完全交给模型。这保持了教学上的"最小完整"，但研究文档考点 2 明确说"终止条件与最大步数是工程必须项"，考点 10 也警告链式衰减（单步 95% × 20 步 ≈ 36%）。真实 Claude Code 的循环外有权限门、上下文压缩（compaction）、预算控制层层兜底——本仓库是渐进式的：预算这个问题要到 s05 的 Stop hook / s17 goal loop 才以最小形态补上。第二个点是 `run_bash` 的 50000 字符截断与 `"(no output)"`：这其实是**上下文工程（考点 6）的雏形**——工具输出也是上下文，不裁剪就是 context rot 的直接来源；空串和 `"(no output)"` 对模型的可解释性差异很大，前者像"成功但静默"，后者给了模型一个可推理的观察。第三，`tool_result` 回 user 角色而非 system——中途插 system 消息在 API 层不合法，而且把观察放在"人类世界的反馈"位置，正是 ReAct 的 Observation 槽位。**替换实验的心理预期**：把 while 删掉，产品立刻退化为单轮 chatbot；把截断删掉，一句 `cat big.log` 就能灌爆上下文让后续所有轮次退化。

### 4.【面试考点映射】

对应考点 1（Agent/Chatbot/Workflow 区别）、考点 2（Agent Loop 完整循环）、考点 3（ReAct 为何有效）、考点 19（模型是地板、harness 是天花板——核心循环只有 30 行，价值全在周边工程）。30 秒答题框架：先给一句话定义（`while stop_reason == "tool_use"` 循环）→ 拆四要素（模型决策/工具执行在客户端/观察回填/终止条件三分类：模型自停、预算步数、人工中断）→ 补工程要点（观察截断、错误也要回填、步数上限）。两个常见追问：①"为什么 tool_result 要回填到 user 角色？"——API 角色约束 + Observation 语义上就是"环境给用户新的输入"；②"这么干净的循环缺什么才能上生产？"——权限门（s03）、hook（s04）、上下文压缩（s08 主题）、错误恢复与预算（ s05/s17），按需展开。

### 5.【动手实验】

在 `s01_agent_loop/code.py:97` 之后插入一个计数器验证链式衰减直觉：

```python
    turn = 0
    while True:
        turn += 1
        if turn > 10:
            return  # 强制终止，模拟 max_turns 预算
```

跑一个"给当前目录写一个冒泡排序并测试"的 8+ 步任务，观察第 10 步被截断后用什么信号收尾；再把上限改成 20，统计两个设置下的任务成功率——你会直观看到"给模型配的步数预算，就是它可靠性的上限"。

---

## s02 Tool Use：循环不变，工具层生长

### 1.【定位与动机】

s01 只有一个 bash。让模型用 `echo >` 拼写文件、靠 `ls` 猜结构，既慢又易错，而且 bash 的自由度让安全边界无从谈起。s02 引入 read_file / write_file / edit_file / glob 四个语义工具加一张分发表，证明 harness 扩展的**正确杠杆点在工具层而非循环层**：约 150 行新代码，`agent_loop` 形状零改动。对应考点 4"Function Calling 的完整原理"——工具集以 JSON Schema 描述、模型只生成结构化调用请求、**执行永远发生在客户端（harness）侧**、结果作为 tool message 回填——本 stage 是这套机制在 Python 里的最小写实。

### 2.【源码精读】

分发表是新核心（`s02_tool_use/code.py:151-156`）：

```python
# -- New in s02: dispatch map (replaces s01's hard-coded run_bash call) --
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}
```

`code.py:159-161` 的注释把演进点题得很直白：s01 是 `output = run_bash(block.input["command"])` 的硬编码分支，s02 变成 `output = TOOL_HANDLERS[block.name](**block.input)` 的一次字典查表。执行处（`code.py:178-183`）有保底：`handler = TOOL_HANDLERS.get(block.name)`，查不到时输出 `"Unknown: {name}"`——**未知工具不崩循环，而是把错误文本喂回模型让它自纠**。`safe_path`（`code.py:81-85`）是文件工具的安全地基：

```python
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
```

先 `resolve()` 归一化（消解 `..`、符号链接），再 `is_relative_to` 判断是否仍在工作区内。`run_edit`（`code.py:108-117`）值得逐行看：`old_text not in text` 时返回 `"Error: text not found"`（失败信号回给模型），命中则 `text.replace(old_text, new_text, 1)` 只替换第一次出现。`run_glob`（`code.py:120-133`）做了两次防护：结果逐条 `is_relative_to` 过滤（glob 独立展开路径，`../` 有可能混进来），且 200 条封顶、超限提示"narrow the pattern"。

### 3.【设计权衡】

为什么把 read/write/edit 从 bash 里独立出来？三个理由：①**结构化 schema 参数**（`path`、`content`、`limit` 是 JSON 对象）比对 shell 转义引号可靠得多，工具描述就是给模型的"API 文档"，描述质量直接决定调用准确率；②**原子语义**——edit 给了模型"精确替换一次"的原子操作，比 sed 的正则替换失败模式温和；③**分级授权的前提**——file 工具与 bash 分开后，s03 才能对"写操作越界"做规则化人审，而对 bash 单独走 deny list。与真实 Claude Code 对照：Read/Write/Edit/Glob 独立于 Bash 的工具面结构同构；差异是真实产品的 Edit 对 old_string 唯一性有更严的约束与更聪明的失败提示，本教学版把"多次出现时静默替换第一次"留作已知简化。**如果删掉 dispatch map**，每加一个工具都要给循环打 if/elif 补丁，工具集合与控制流重新耦合，正是这个 stage 要消灭的坏味道。**如果删掉 safe_path**，`write_file` 一个绝对路径就能改工作区外的文件——注意 `run_glob` 的二次过滤说明"路径校验要在每个产生路径的入口各做一次"。

### 4.【面试考点映射】

对应考点 4（function calling 全链路：schema 声明 → 模型输出名称+参数 → 客户端 dispatch 执行 → tool message 回填；并行调用天然体现在"一次响应多个 tool_use 块"）、考点 14（成本：`read limit` 参数和 50k 截断都是 token 预算手段）、考点 19（同等模型下，工具 schema 与错误信息质量的差异大于换模型）。30 秒答题框架："模型产 JSON、harness 跑函数、结果回对话"三句话 + 指出约束解码保证格式 + 强调执行不在模型侧。追问 1：工具描述写不好会怎样？——参数幻觉、该调不调、错位调用，等价于"API 文档写烂的库没人用对"。追问 2：handler 内部抛异常怎么办？——必须原地捕获并转成错误文本回填（本版 run_* 已这么做，`agent_loop` 里 `handler(**input)` 若抛裸异常会击穿整个 REPL），错误是给模型看的输入，不是给用户看的崩溃。

### 5.【动手实验】

两处小改各验证一个假设：①在 `code.py:114` 前加 `cnt = text.count(old_text)`，当 `cnt != 1` 时返回一个 dict 提示 "+{cnt-1} more matches; make old_text unique"——复刻真实 Edit 的唯一性校验，观察含重复片段文件的编辑成功率变化；②在 `TOOLS` 数组里把 `run_edit` 的 description 从一句话加长为"必须先 read 文件再 edit"——量化工具描述措辞对调用准确率的影响，这直接演示考点 19 的"工具描述质量决定 harness 天花板"。

---

## s03 Permission：执行前加三道闸门

### 1.【定位与动机】

s02 的 agent 已经有 5 个工具，但 bash 除了 `run_bash` 里那个内联 deny list 别无约束。用户一句"帮我清理一下项目"，模型可能给出 `rm -rf /`——安全边界必须由代码负责，且判断必须发生在工具执行**之前**，而不是执行函数内部。s03 建立"硬拒绝 → 规则匹配 → 人工审批"三道闸门，直接对应研究文档考点 13（沙箱与权限设计：默认只读、写与网络按需授予、四档权限模式）与考点 18（HITL：分级审批、interrupt-on-external-effect）。这也是研究文档"评测重心从能力分转向安全与过程分"里最基础的"过程分"设施。

### 2.【源码精读】

三闸门的串联逻辑 `check_permission`（`s03_permission/code.py:203-214`）：

```python
def check_permission(block) -> bool:
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m[blocked] {reason}\033[0m")
            return False
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False
    return True
```

**Gate 1 硬拒绝**：`DENY_LIST`（`code.py:158-164`）是 `rm -rf /`、`sudo`、`shutdown`、`mkfs`、`dd if=` 这样的无条件禁区，命中即拒、不商量。**Gate 2 规则匹配**：规则数组 `PERMISSION_RULES`（`code.py:177-185`）每条是 `{tools, check(args), message}` 三元组，两条规则分别拦截"文件路径越出工作区"和"潜在破坏性 bash"。破坏性命令的判定（`code.py:168-170`）是个讲究的正则：

```python
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
```

`(?:^|[;&|()\n])` 要求 rm/del 前是命令分隔符（行首|分号|管道|括号），`(?=\s|$|...)` 要求后面是空白或分隔符——避免 `confirm`、`form` 这类单词被子串误伤。**Gate 3 人审**：`ask_user`（`code.py:195-199`）阻塞式 `input("Allow? [y/N]")`，默认 N。 notice `SYSTEM`（`code.py:66-71`）同时变了：加进 "Do not ask for approval yourself… the harness will approve or deny before execution"——把审批职责**从 prompt 层剥离到 harness 层**。循环内的接线只有一处（`code.py:237-241`）：拒绝时向模型回填 `tool_result "Permission denied."` 后 `continue`。

### 3.【设计权衡】

教学版实现的是研究文档"四档权限模式"的**雏形映射**：`DENY_LIST`+规则默认放行 ≈ 自动授权档；规则命中进人审 ≈ "完全自动+白名单"的反面配置；若给所有写操作都加规则则逼近"人审全批"档；若把 bash/write 全 deny 只留 read/glob 就是"只读档"——四档 = 同一条管线上"规则命中后走 ask 还是走 pass"的配置差异，而不是四套代码。与真实 Claude Code 的差距：真实产品允许中途中切换权限模式、规则以配置文件声明（allow 列表可持久化"本命令不再询问"），教学版的审批是每次都问的**无记忆审批**，既打断心流又训练用户盲按 y——把 ask 改成"本会话记住"是本文实验题。另一个关键设计：**拒绝也回填 tool_result 而不是抛异常**。错误信息是上下文工程的一部分（考点 12 防御纵深里"输出检查"的反向应用）：模型看到 denied 能改道（比如改用更安全的命令），好于循环崩溃或者模型困惑重试。deny list 的子串匹配是已知简化——`echo "sudo is bad"` 会误伤、`rm -fr /` 变形可绕过——真实产品用更精细的解析与规则库，这个差距本身就是面试谈资。

### 4.【面试考点映射】

对应考点 13（权限系统设计五要点：默认只读/写与网络按需/容器与 worktree/凭证分权/审计——本 stage 覆盖前三点的地基：安全判断前置、结构化参数可解析、拒绝可记录）、考点 18（分级审批设计与"可回滚是敢于放权的前提"）、考点 12（prompt injection 时权限最小化是最后防线：带着注入的网页骗模型调 write_file 出工作区，会被 Gate 2 的人审拦下）。30 秒答题框架：默认只读 → 三闸门（无条件 deny/上下文规则/人审）→ 拒绝也回填保持循环活性 → 同一管线上配置出四档权限模式。追问 1：权限判断放模型 prompt 里自查行不行？——不行，模型本身就是攻击面，被注入后"自己批准自己"，判断必须在模型输出之后、工具执行之前、由不变代码完成。追问 2：怎么实现"同一操作本会话不再询问"？——按 (tool, 规则指纹) 记录已批集合，命中规则先查会话记忆，并讨论是否落盘为 allow 规则（会话级 vs 项目级持久化的差别）。

### 5.【动手实验】

在 `code.py:194`（`ask_user` 定义前）加一个会话级批准缓存，共三行改动验证"持久ized 授权"假设：

```python
APPROVED = set()
def ask_user(tool_name, args, reason):
    key = f"{tool_name}:{reason}"
    if key in APPROVED:
        return "allow"
    choice = input("   Allow? [y/N] ").strip().lower()
    if choice in ("y", "yes"):
        APPROVED.add(key)
        return "allow"
    return "deny"
```

同一长任务里第二次触发同类规则时应不再询问；再思考一个开放题：`APPROVED` 若写进 `~/.config` 就是持久白名单——四档中的"全自动+白名单"档由此成型。

---

## s04 Hooks：把循环固定，把策略外置

### 1.【定位与动机】

s03 顺手暴露了一个架构问题：权限逻辑直接写死在循环里（`check_permission`），每加一条策略（日志、限流、内容扫描）都要改 `agent_loop`。s04 回答的是 harness 的**可维护性与可扩展性**问题：循环一次定稿，所有扩展逻辑走固定生命周期事件——UserPromptSubmit（提示词提交时）、PreToolUse（工具执行前）、PostToolUse（工具执行后）、Stop（循环将退出时）。s03 的全部权限逻辑原样平移成一个 PreToolUse 钩子，并新增日志/大输出/总结钩子。这是研究文档"harness 工程是天花板"论点（考点 19）的工程面，也是真实 Claude Code"hooks 不可绕过"特性的最小模型。

### 2.【源码精读】

钩子机制三件套（`s04_hooks/code.py:137-147`）：

```python
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # A hook result blocks this tool call.
            return result
    return None
```

约定：**钩子返回非 None 即拦截**，依次执行、首个拦截者生效。s03 的 `check_permission` 原封不动变成 `permission_hook`（`code.py:162-186`，deny list→人审的两层结构保留），注册表在 `code.py:213-217`（五个钩子按序挂上）。循环里有两处消费：PreToolUse 拦截点（`code.py:245-249`，被拦截同样回填 tool_result、模型可改道）与 PostToolUse（`code.py:254`，只观察不拦截）。最微妙的是 Stop 钩子（`code.py:235-240`）：

```python
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return
```

模型本想结束，Stop 钩子返回一个字符串时，它被当作 user 消息追加进对话然后 `continue`——**钩子可以改变控制流，"强迫模型继续"**。这是 s05 催更与后续 goal loop 的机制前身。

### 3.【设计权衡】

"不可绕过"的本质在第 245 行：`trigger_hooks("PreToolUse", block)` 是循环主路径上的一段代码，不是模型可选的服务。模型只能决定"调什么工具"，只要它调工具，就必然先穿过所有已注册的 PreToolUse 钩子；就算它直接说文本想结束，也要过 Stop 的关卡。对比 s03 的 `if not check_permission(block)`——行为等价，但控制反转后权限策略与执行引擎解耦：添加新策略只 register 一行，不改循环。与真实 Claude Code 的机制差异：真实 hooks 以 shell 命令配置在项目/用户配置中、由 harness 在对应生命周期点跨进程调用，模型既无法不执行也无法篡改；教学版是进程内 Python 回调，同样的不可绕过性，更少一层序列化。**裁撤后果**：删掉 hook 系统，一切回到 s03 模式——每个安全/审计需求都是一次对循环的 diff，长此以往循环变成意大利面；而钩子返回值若没有"非 None 即拦截"的统一语义，就无法在一个循环里表达"拦截型"与"观察型"两种钩子。PostToolUse 可攻可守的边界也值得注意：工具已执行、输出已生成，钩子只能记录告警（large_output_hook, `code.py:194-198`），真正的"写后回滚"需要 snapshot 机制支撑，这里留白。

### 4.【面试考点映射】

对应考点 19（harness 工程质量：同等模型下，扩展点设计决定团队能往上叠多少层策略）、考点 18（PreToolUse 就是分级审批的正确执行点）、考点 12（注入防御纵深中"输出检查"在 PostToolUse 落点）。30 秒答题框架：固定循环 + 生命周期事件 → 钩子运行在 harness 内所以模型不可绕过 → PreToolUse 可拦截 / PostToolUse 观察 / Stop 可强制继续 → 真实产品里这些钩子是配置化的外部命令。追问 1：钩子抛异常是放行（fail-open）还是拦截（fail-closed）？——安全钩子必须 fail-closed（异常当作拦截并回填原因），日志钩子可以 fail-open，这暴露了单一 trigger 语义的不足、要升级为按钩子声明容错策略。追问 2：Stop 钩子返回字符串强制继续，怎么防止它造成死循环？——加"强制次数预算"，连续 N 次 force 后转为真终止（可现场口说 patch：在 `agent_loop` 加 `force_count`）。

### 5.【动手实验】

十行内验证两个拦截假设：①新增一个敏感文件保护钩子并注册最前（`code.py:217` 前）：

```python
def protected_paths_hook(block):
    p = str(block.input.get("path", ""))
    if p.endswith((".env", "secrets.py")) or "/.git/" in p:
        return "Protected path (secrets/refs) — use a different file"
register_hook("PreToolUse", protected_paths_hook)
```

②把 `summary_hook`（`code.py:206-211`）改成：当 session 工具调用数 > 8 时返回 `"Wrap up now: summarize what you did and stop."`——用 Stop 钩子实现最小预算控制，跑一个长任务看模型如何收尾。

---

## s05 TodoWrite：给模型一副任务状态机

### 1.【定位与动机】

s01–s04 的模型能干活，但多步任务里有两个老毛病：**埋头苦干忘了计划**（做到第 7 步时当初的计划早已被稀释在上下文里——正是研究文档 context rot 的过程化表现）以及**用户看不到进度**。s05 给模型一副结构化任务板：新增 `todo_write` 工具 + `TodoManager` 状态机（pending / in_progress / completed 三态），外加一个 harness 侧的"催更"机制——连续 3 轮没更新 todo 就在工具结果旁边注入 `<reminder>`。对应考点 17"规划怎么做"（任务分解 + 执行中重规划）、考点 2（循环里的进度可观测），精简版 `agents/s03_todo_write.py:2` 的注释把哲学说透了："keeping the model on course **without scripting the route**"——计划由模型自管，harness 只递话筒，不抢方向盘。

### 2.【源码精读】

状态机的校验层（`s05_todo_write/code.py:140-160`）是这个 stage 的精华：

```python
        validated = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] must be an object")
            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] has invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"content": content, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")
```

五连校验：≤20 条上限（`code.py:137-138`）、每条必须是 object、content 非空、status 枚举封闭、**同一时刻最多一个 in_progress**（`code.py:156-157`）——单焦点约束保证"正在做什么"永远可读。`render()`（`code.py:162-177`）渲染成 `[ ] / [>] / [x]` 三态符号加 `(n/m completed)` 进度行，这个字符串同时给两个受众：模型（下轮上下文）与人（终端打印）。双层容错在 `run_todo_write`（`code.py:183-189`）：schema 层已经声明 `maxItems: 20`、`minLength: 1`、status enum（`code.py:203-204`），穿透 schema 的坏输入再被 `try/except ValueError` 转成 `"Error: ..."` 文本回填——**错误也是给模型的输入**。催更引擎在 `agent_loop`（`code.py:317-344`）：

```python
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.append({"type": "text",
                            "text": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0
```

本轮只要执行过 `todo_write`（`code.py:334-335` 置位 `used_todo`）计数就归零；3 轮未更新，则在 tool_result **同一个 user 回合里**追加一个独立 text block，注入后归零重计。SYSTEM prompt（`code.py:56-59`）同步加了"Before starting any multi-step task, use todo_write to plan your steps"。

### 3.【设计权衡】

三个层面的取舍：①**nudge 而非硬路由**——真实 Claude Code 的 TodoWrite 同样是"提醒模型更新"而非 harness 代写计划；一旦 harness 写死执行顺序，系统就从 Agent 滑向 Workflow（考点 1 的边界），s05 的设计保住了这一点：模型可以选择忽略 reminder（代价只是反复被催），控制流自主权仍在模型。真实产品的做法更进一步——reminder 以 `<system-reminder>` 形式携带完整任务状态注入，本教学版的固定文案是简化。②**全量重写 vs 增量字段**——`update()` 每次接收整张清单（与真实 TodoWrite 口径一致）：对 LLM 友好（无需维护自己的 ID 账本、天然收敛状态），代价是 token 略多；精简版 `agents/s03_todo_write.py:68-70` 的记录带 `id/text` 字段、顶层版是 `content`，两版字段差异说明这是可漂移的实现细节，状态机三态才是稳定接口。③**单 in_progress 约束**——删掉的后果不是报错而是状态板失真：模型开始"假装并行"，用户与后续子代理都失去"当前焦点"这一个最重要的信号。若彻底删掉 `todo_write`：计划退化成潜谈文本，多步任务后半段丢步骤率高——这正是催更机制存在的证据。

### 4.【面试考点映射】

对应考点 17（规划谱系：CoT/分解/反思里，工程化的 decompose+重规划以工具形态出现——`plan-first` 由 SYSTEM 引导，重规划由 reminder 驱动）、考点 6（reminder 注入是标准的上下文工程手段，和 CLAUDE.md / system-reminder 同源）、考点 9 的铺垫（结构化外部状态是子代理/多智能体 handoff 时不丢上下文的前提，单 agent 先具备"状态可序列化"）。30 秒答题框架：结构化计划用工具而非自由文本 → 三态 + 单焦点状态机 → 校验错误回填让模型自纠 → 3 轮未更就 nudge、保留模型自主权 → 状态板人机同读。追问 1：为什么 reminder 注入在 tool_result 所在的 user 回合、而不是每轮追加 system 消息？——API 不允许会话中途插 system，且紧随工具结果的观察位（ReAct 的 Observation 槽）注意力效率最高。追问 2：催更的 3 轮阈值怎么调？——阈值越小提醒噪声越大（token 浪费、还可能诱导模型刷 todo_write），越大越容易任务漂移；3 是"一个任务段落"的工程拍数，正确答案应该跟任务粒度和上下文压缩频率联动。

### 5.【动手实验】

两处改动验证两个假设：①把 `code.py:341` 的 `>= 3` 改成 `>= 1` 再跑多步任务，记录 token 增量与模型行为（应观察到提醒疲劳与频繁刷状态）；②把注入的 reminder 升级为携带状态的完整版：

```python
        if rounds_since_todo >= 3:
            results.append({"type": "text",
                            "text": f"<reminder>Update your todos. Current:\n{TODO.render()}</reminder>"})
            rounds_since_todo = 0
```

观察模型在"看得见当前清单"时是否更容易纠正漂移——这就是从教学版到真实 `<system-reminder>` 语义的最后一公里。

---

## 面试火力图（本组五连）

| Stage | 直接命中的考点 | 火力 | 一句话火力点 |
|---|---|---|---|
| s01 Agent Loop | 考点 2（Agent Loop）· 考点 1（Agent vs Chatbot vs Workflow）· 考点 3（ReAct）· 考点 19（harness 天花板） | ⭐⭐⭐⭐⭐ | `while stop_reason == "tool_use"` 一句话背下 agent 的全部秘密；终止条件三分类（模型自停/预算/人工）是加分副炮 |
| s02 Tool Use | 考点 4（Function Calling）· 考点 14（成本截断）· 考点 19（工具设计差异） | ⭐⭐⭐⭐⭐ | "模型产 JSON、harness 跑函数、结果回对话"；dispatch map 让循环零改动生长出 5 个工具 |
| s03 Permission | 考点 13（权限与沙箱）· 考点 18（HITL 分级审批）· 考点 12（注入防御纵深） | ⭐⭐⭐⭐⭐ | 三闸门 = 硬 deny/规则/人审；四档权限模式是同一条管线上的配置差异，不是四套代码 |
| s04 Hooks | 考点 19（harness 工程质量）· 考点 18· 考点 12 | ⭐⭐⭐⭐ | 扩展逻辑从循环里外置到生命周期事件；`trigger_hooks` 硬编码在主路径 ⇒ 模型**无法绕过**；Stop 钩子能改控制流 |
| s05 TodoWrite | 考点 17（规划与重规划）· 考点 6（上下文工程/nudge 注入）· 考点 9（状态可序列化铺垫） | ⭐⭐⭐⭐ | 三态单焦点状态机 + 3 轮未更注入 reminder：计划归模型、催更归 harness，控制流自主权不丢 |

> 串联一句话：s01 定循环骨架（模型决定控制流），s02 在工具层生长，s03 把安全闸门插进执行前，s04 把一切策略外置成不可绕过的钩子，s05 给模型自管的状态与 harness 的提醒——五步正好覆盖"三层知识树"的编排层主线，火力最密处（s01/s02/s03）也是考点 Top20 中五星题最集中的地方。
