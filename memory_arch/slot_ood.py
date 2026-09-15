"""Out-of-template Chinese phrasings for slot extraction.

The shared Python extractor lives in ``igm/gate.py`` and
``IGMMethod._extract_slot`` delegates to it; the nested JavaScript DSH fixture
is held to the same oracle by parity tests. The synthetic corpus in
synth_data.py generates facts as "我的{attr}是{value}" and questions as
"我现在的{attr}是什么？" — the rule's own templates — so no load built from it
can show what the rule does to phrasing nobody authored on purpose.

This module is that missing set: utterances an agent actually meets, labelled
by whether they name a durable attribute.  A wrong label is not cosmetic,
because a non-None slot arms supersede (igm/store.py), so a spurious key lets
one write delete an unrelated memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SlotCase:
    text: str
    expect: str | None      # None = must not extract; str = exact key required
    why: str


@dataclass(frozen=True)
class UpdateChain:
    old: str
    new: str
    attribute: str
    defect: str | None      # None for the control case (phrased on-template)


@dataclass(frozen=True)
class CollisionPair:
    first: str
    second: str
    shared_slot: str        # the key both utterances wrongly map to


# Speech acts that are not attributes.  Interjections and quoted stances all
# sit in "我的{X}是" shape, which is exactly what the rule keys on.
NOT_ATTRIBUTE = [
    SlotCase('我的天，这个项目的部署流程是手动跑三个脚本。', None, '感叹语，"我的天"不是属性'),
    SlotCase('我的天哪，网关准备用 Rust 重写。', None, '感叹语'),
    SlotCase('我的意思是这里应该用毫秒而不是秒。', None, '"我的意思是"是解释措辞，不是属性'),
    SlotCase('我的意思是他先提的是这个方案。', None, '同上，且与上一条话题无关'),
    SlotCase('我说的就是这个意思，你先按方案 A 走。', None, '指涉前文，无属性'),
    SlotCase('我让你改的是配置文件，不是启动脚本。', None, '纠正指令，无属性'),
    SlotCase('我现在不方便，晚点再联系。', None, '瞬时状态，不是长期事实'),
    SlotCase('我去过一次上海，那次演讲是临时安排的。', None, '过去事件叙述，非当前状态属性'),
    SlotCase('我在想这个测试要不要拆成两个文件。', None, '思考中的问题'),
    SlotCase('我就知道这个 CI 会挂。', None, '评论'),
]

# Attributes named in the sentence but phrased off-template.
NAMED_ATTRIBUTE = [
    SlotCase('我的工号是 A1024。', '工号', '系词 + 短属性名'),
    SlotCase('我的生日是 3 月 12 日。', '生日', '系词 + 短属性名'),
    SlotCase('我的导师是李老师。', '导师', '系词 + 短属性名'),
    SlotCase('我的座位号是 3 排 12 座。', '座位号', '系词 + 属性名带后缀'),
    SlotCase('我的周报时间是每周四下午。', '周报时间', '属性名后紧跟系词'),
    SlotCase('我的账号等级是 L3，上个月升的。', '账号等级', '属性值后接从句'),
    SlotCase('我用的编辑器是 VS Code。', '编辑器', '关系从句作定语，只应取中心词'),
    SlotCase('我的邮箱改成 new@example.com 了。', '邮箱', '更新动词"改成"，句中无系词'),
    SlotCase('我把主分支改成 develop 了。', '主分支', '"把…改成…"处置式'),
    SlotCase('我的手机号的运营商从联通换成了移动。', '运营商', '属性名在第二个"的"之后'),
    SlotCase('我在公司的工位搬到 5 层东侧了。', '工位', '"搬到"，且属性名在处所状语后'),
    SlotCase('我的 GitHub 用户名是 stringadmin。', 'GitHub用户名', '中英混排属性名超出长度上限'),
    SlotCase('我的项目代号叫 FIP，全称是特征重要性保护。', '项目代号', '命名动词"叫"'),
]

# Write pairs: the second utterance updates the same attribute, so the store
# must end up holding one current value.  defect names why it may not.
UPDATE_CHAINS = [
    UpdateChain('我的住址是北京。', '更新一下，我的住址现在是深圳了。', '住址', None),
    UpdateChain('我的邮箱是 old@example.com。', '我的邮箱改成 new@example.com 了。', '邮箱',
                '非系词更新动词漏抽 slot，旧值不被覆盖'),
    UpdateChain('我的工位是 3 层东侧。', '我在公司的工位搬到 5 层东侧了。', '工位',
                '同上：漏抽后旧值与新值并存'),
    UpdateChain('我的常驻城市是杭州。', '我的常驻城市目前是成都。', '常驻城市',
                '时态修饰词只在属性名开头剥离，残留后键不稳定'),
]

# Unrelated facts whose extracted keys collide; both must survive.
COLLISION_PAIRS = [
    CollisionPair('我的意思是这里应该用毫秒而不是秒。', '我的意思是他先提的是这个方案。', '意思'),
    CollisionPair('我的想法是做一版最小的原型先看效果。', '我的想法是周末先把 CI 补上。', '想法'),
]


def evaluate(extract: Callable[[str], str | None]) -> dict:
    """Score one slot-extraction implementation against the labelled corpus."""
    false_extractions = [
        {"text": c.text, "slot": extract(c.text), "why": c.why}
        for c in NOT_ATTRIBUTE if extract(c.text) is not None
    ]
    wrong_keys = [
        {"text": c.text, "expected": c.expect, "got": extract(c.text), "why": c.why}
        for c in NAMED_ATTRIBUTE if extract(c.text) != c.expect
    ]
    return {
        "not_attribute_total": len(NOT_ATTRIBUTE),
        "not_attribute_extracted": len(false_extractions),
        "false_extraction_rate": round(len(false_extractions) / len(NOT_ATTRIBUTE), 4),
        "named_total": len(NAMED_ATTRIBUTE),
        "named_exact": len(NAMED_ATTRIBUTE) - len(wrong_keys),
        "named_key_accuracy": round((len(NAMED_ATTRIBUTE) - len(wrong_keys)) / len(NAMED_ATTRIBUTE), 4),
        "false_extractions": false_extractions,
        "wrong_keys": wrong_keys,
    }
