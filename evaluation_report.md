# 当前模型训练质量评估

结论：对当前语料的字符预测拟合很强，日常对白形式已有学习；回答相关性、指令遵循和上下文信息利用仍明显不足。验证集高度重复，使低验证损失不足以证明泛化能力。

## 评估对象与方法

- 权重：`model_final.pth`，参数量 10,797,388，字符词表 76。
- 评估时未发现 `model_bpe.pth`，本报告不代表 BPE 语言模型的训练质量。
- 训练集与验证集分别抽取 256 个固定随机种子的窗口，每窗口预测 256 个后续字符，共各 65,536 次预测；窗口可能重叠。
- eval 模式、FP32、关闭 dropout；使用与训练相同的随机窗口协议。该数值不是完整长文本滑动窗口困惑度。
- 只读取已有权重并做前向评估，未修改权重或模型源码。

## 拟合指标

| 指标 | 训练集 | 验证集 |
|---|---:|---:|
| 交叉熵（nats/字符） | 0.0912 | 0.0989 |
| 字符级困惑度 | 1.0955 | 1.1039 |
| 下一个字符预测准确率 | 97.04% | 96.84% |
| bits/字符 | 0.1316 | 0.1426 |

字符准确率不是问答准确率。训练与验证损失差距小，当前抽样未显示典型的大幅训练/验证差距；数据重叠使这一结论不能用于排除记忆或过拟合风险。

## 验证集独立性

- 验证集 6,301 条非空对白行中，5,952 条在训练集中逐字匹配（去掉行首尾空白），占 94.46%。
- 完整对话层面，792 段验证对话中没有与训练对话完全相同的段落（忽略大小写与空白差异）；大量重叠发生在对白行和模板组合层面。
- 本次抽样中，未在训练集出现过的对白行对应 4,944 次字符预测，其平均损失为 0.2127；重复对白行为 0.0896。
- 完整对白长度中位数为 475 字符；全部完整对白都超过旧模型的 256 字符上下文。

## 实际生成样本

固定温度 0.8，遇换行截取当前一轮；前 8 项使用随机种子 1337。回复在训练文本中出现不单独证明记忆，但与大量模板重叠、答非所问结合，构成明显的训练句子复用迹象。

| 测试 | 提问或上下文 | 实测回复 |
|---|---|---|
| greeting（seed 1337） | Hello? | Oh my gosh, hey! It's been what, like a year? |
| identity（seed 1337） | Who are you? | Absolutely good, have a seat. Do you have a photo ID with you? |
| arithmetic（seed 1337） | What is 2 + 2? | I'm so in. How long were you thinking? A weekend or the full week? |
| fact（seed 1337） | What is the capital of France? | If it keeps up, might be worth seeing Dr. Kim about it. |
| instruction（seed 1337） | Say exactly: banana. | You're a lifesaver, Emma. Coffee's on me. |
| daily_advice（seed 1337） | I'm feeling tired today. What should I do? | Rest, plenty of fluids, and give it a few days. |
| daily_request（seed 1337） | Can you help me move this weekend? | I can do noon that see how it feels. They look great. |
| daily_problem（seed 1337） | I lost my keys. What should I do? | Rest, plenty of fluids, and give it a few days. |
| short_context_memory（seed 1337） | A: My name is Alex. / B: Nice to meet you, Alex. / A: What is my name? / B: | Works for me. Should I bring anything? |
| alternate_seed（seed 2026） | Who are you? | The tram fally dog's in the analytics portal and class starts at 3 o'clock. |
| alternate_seed（seed 2026） | What is 2 + 2? | I'm so in. How long were you thinking? A weekend or the full week? |

前 8 条单轮测试中，有 6 条回复全文（至少 20 个字符）能在训练文本中找到。问候和疲劳建议有合理回应；算术、首都、精确输出指令、丢钥匙建议及记住名字等测试未成功。这是小样本定性检查，不能当作总体任务准确率。

## 优先改进方向

1. 先按完整对话的相似度和重复对白行审计、整理数据，再划分训练/验证/测试集。保持对话完整，避免简单删除句子破坏问答关系。
2. 增加多样且真正对应问题的回答数据，以及一个与训练模板不同的固定人工测试集。
3. 记录训练步数、训练/验证曲线，保存验证表现最好的权重；目前只有最终权重，无法判断是否早已收敛或中途退化。
4. 新 BPE 模型产出后使用同一组任务问题复测。字符与 BPE 的 loss/PPL 单位不同，不能直接横向比较数值。

## 配置检查

`config.py` 中 `TrainConfig.max_iters = 2000` 当前未接入训练入口；`train.py` 实际仍使用 `max_iters = 5000`。checkpoint 也未保存真实已完成步数，因此不能仅凭配置推断旧权重训练了多少步。

## 参考资料

- [训练数据去重与训练/验证重叠的研究](https://aclanthology.org/2022.acl-long.577/)
- [困惑度与 tokenizer、上下文长度的关系](https://huggingface.co/docs/transformers/perplexity)
