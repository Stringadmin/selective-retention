"""BGE validation for RAG context routing plus outcome-backed experience.

The workload uses held-out Chinese phrasings for operational situations. RAG
maps a new phrasing to a canonical context ID; OutcomeMemory then uses feedback
attached to that ID. This is intentionally a component test, not an end-to-end
LLM agent benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from . import Embedder
from .context_router import ContextRoute, SemanticContextRouter
from .outcome import OutcomeMemory


ACTIONS = ("plan_a", "plan_b")

# Prototypes are the only texts visible to the router's index. Every query
# below is held out from that index; they are hand-authored, limited probes.
CONTEXTS: dict[str, dict[str, object]] = {
    "release_failure": {
        "prototype": "线上发布后服务报错，需要处理发布风险。",
        "queries": [
            "新版上线以后接口持续返回 500。",
            "刚部署的服务已经不能正常访问。",
            "生产环境更新完成后出现大量请求异常。",
            "代码发布之后用户访问功能失败。",
        ],
    },
    "schema_migration": {
        "prototype": "数据库结构迁移失败，需要恢复数据一致性。",
        "queries": [
            "执行 schema 变更时数据库报错了。",
            "表结构升级没有完成，数据状态不一致。",
            "迁移脚本中断后需要处理数据库。",
            "数据库改字段失败，担心已有记录受影响。",
        ],
    },
    "payment_anomaly": {
        "prototype": "支付扣款出现异常，需要核对交易状态。",
        "queries": [
            "用户付款后订单仍显示未支付。",
            "收款流程可能重复扣费了。",
            "支付回调异常，交易状态对不上。",
            "客户说钱扣了但订单没有成功。",
        ],
    },
    "secret_exposure": {
        "prototype": "发现访问密钥可能泄露，需要处理安全事件。",
        "queries": [
            "代码仓库里疑似提交了 API 密钥。",
            "监控发现凭证可能被外部人拿到。",
            "有人把生产 token 暴露在日志中。",
            "怀疑服务账号的密码已经泄漏。",
        ],
    },
    "latency_incident": {
        "prototype": "线上服务响应延迟升高，需要排查性能故障。",
        "queries": [
            "用户反馈页面加载突然非常慢。",
            "接口 P95 延迟持续上涨。",
            "请求处理卡顿，服务响应时间异常。",
            "线上系统开始频繁超时。",
        ],
    },
    "cache_inconsistency": {
        "prototype": "缓存与源数据不一致，需要处理陈旧结果。",
        "queries": [
            "用户看到的页面还是旧数据。",
            "更新记录后缓存没有同步变化。",
            "读取到的内容和数据库最新值不一样。",
            "缓存似乎返回了过期信息。",
        ],
    },
    "disk_exhaustion": {
        "prototype": "服务器磁盘空间耗尽，需要恢复运行容量。",
        "queries": [
            "机器的硬盘快满了，服务无法写入文件。",
            "日志把磁盘占满，程序开始报空间不足。",
            "服务器没有剩余存储容量了。",
            "容器写数据时报 no space left。",
        ],
    },
    "login_failure": {
        "prototype": "用户无法登录账号，需要排查认证故障。",
        "queries": [
            "许多用户输入正确密码也进不了系统。",
            "登录接口开始拒绝正常账号。",
            "认证服务异常，用户会话无法建立。",
            "客户反映账号登录一直失败。",
        ],
    },
}

# These queries were written after inspecting the calibration routes. They are
# held out from both the router index and threshold selection. This reduces one
# source of selection bias, but remains a small hand-authored component test.
EVALUATION_QUERIES: dict[str, tuple[str, ...]] = {
    "release_failure": (
        "线上版本切换后 API 大量报错。",
        "新功能部署到生产后服务异常中断。",
        "发布操作完成后用户请求失败。",
        "回滚前需要确认本次上线导致的故障。",
    ),
    "schema_migration": (
        "数据库迁移任务失败，部分表没有更新。",
        "DDL 执行中断，需要核验数据是否一致。",
        "升级表字段后发现迁移没有完成。",
        "数据库版本变更留下了不一致的记录。",
    ),
    "payment_anomaly": (
        "订单支付成功但后台没有更新状态。",
        "用户反馈信用卡被扣了两次。",
        "交易通知没有处理，付款和订单不一致。",
        "对账发现支付结果与订单状态冲突。",
    ),
    "secret_exposure": (
        "配置文件误传了云服务访问凭证。",
        "Git 提交中出现疑似私钥内容。",
        "生产密钥可能已被泄露到公开日志。",
        "需要轮换一个被怀疑暴露的服务凭证。",
    ),
    "latency_incident": (
        "服务响应越来越慢，很多请求超时。",
        "API 的 99 分位耗时异常升高。",
        "高峰期页面打开缓慢且请求堆积。",
        "应用处理请求的延迟显著变大。",
    ),
    "cache_inconsistency": (
        "修改数据后客户端依然拿到旧结果。",
        "缓存没有失效，页面显示陈旧内容。",
        "数据库已更新但读接口返回旧值。",
        "用户看到的内容和最新记录不同，像是缓存问题。",
    ),
    "disk_exhaustion": (
        "节点存储空间告警，写入操作失败。",
        "磁盘占用达到上限，需要清理容量。",
        "应用无法落盘，系统提示空间不足。",
        "pod 文件系统已满导致服务写日志失败。",
    ),
    "login_failure": (
        "正常用户凭据无法通过登录验证。",
        "身份认证请求持续失败，无法创建会话。",
        "登录页面接受密码却不让用户进入。",
        "大量账号突然不能登录平台。",
    ),
}


def _queries_for(probe_set: str) -> dict[str, tuple[str, ...]]:
    if probe_set == "calibration":
        return {
            context_id: tuple(spec["queries"])  # type: ignore[arg-type]
            for context_id, spec in CONTEXTS.items()
        }
    if probe_set == "evaluation":
        return EVALUATION_QUERIES
    raise ValueError("probe_set must be 'calibration' or 'evaluation'")


def _populate(memory: OutcomeMemory, rng: random.Random, context_id: str, best_action: str) -> None:
    """Write observed outcomes before the held-out query is issued."""
    for _ in range(8):
        for action in ACTIONS:
            probability = 0.85 if action == best_action else 0.15
            memory.record(
                context_id,
                action,
                rng.random() < probability,
                provenance="verified_simulation",
            )


def _selective_routing(
    routes: dict[str, ContextRoute],
    expected_contexts: dict[str, str],
    min_margin: float,
) -> dict[str, object]:
    """Summarize a fixed-margin selective-routing policy.

    A rejected route intentionally has no automatic action recommendation. The
    caller must send it to a fallback such as a human, a more specific
    retriever, or an explicit clarification step.
    """
    if min_margin < 0:
        raise ValueError("min_margin must be non-negative")

    accepted: list[str] = []
    abstained: list[str] = []
    for query, route in routes.items():
        margin = route.margin
        if margin is None or margin >= min_margin:
            accepted.append(query)
        else:
            abstained.append(query)

    accepted_correct = sum(
        routes[query].context_id == expected_contexts[query]
        for query in accepted
    )
    abstained_incorrect = sum(
        routes[query].context_id != expected_contexts[query]
        for query in abstained
    )
    return {
        "min_margin": min_margin,
        "accepted": len(accepted),
        "abstained": len(abstained),
        "automatic_coverage": len(accepted) / len(routes),
        "accepted_precision": (accepted_correct / len(accepted)) if accepted else None,
        "abstained_incorrect": abstained_incorrect,
        "abstained_examples": [
            {
                "query": query,
                "expected_context_id": expected_contexts[query],
                "context_id": routes[query].context_id,
                "runner_up_context_id": routes[query].runner_up_context_id,
                "margin": routes[query].margin,
                "correct": routes[query].context_id == expected_contexts[query],
            }
            for query in abstained
        ],
    }


def run(
    seeds: int = 500,
    embed_model: str | None = None,
    min_margin: float = 0.02,
    probe_set: str = "calibration",
) -> dict:
    """Evaluate held-out semantic routing and its effect on experience reuse."""
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    prototypes = {
        context_id: str(spec["prototype"])
        for context_id, spec in CONTEXTS.items()
    }
    queries_by_context = _queries_for(probe_set)
    embedder = Embedder(backend="bge", model_path=embed_model)
    router = SemanticContextRouter(embedder, prototypes)
    expected_contexts = {
        query: context_id
        for context_id, queries in queries_by_context.items()
        for query in queries
    }
    routes = {
        query: router.route(query)
        for query in expected_contexts
    }
    route_total = len(expected_contexts)
    route_correct = sum(
        route.context_id == expected_contexts[query]
        for query, route in routes.items()
    )

    exact_correct = 0
    routed_correct = 0
    oracle_correct = 0
    selective_correct = 0
    selective_total = 0
    total = 0
    automatically_accepted = {
        query
        for query, route in routes.items()
        if route.margin is None or route.margin >= min_margin
    }
    event_counts: list[int] = []
    for seed in range(seeds):
        rng = random.Random(seed)
        memory = OutcomeMemory(decay_rate=0.0)
        best_actions = {
            context_id: rng.choice(ACTIONS)
            for context_id in CONTEXTS
        }
        for context_id, best_action in best_actions.items():
            _populate(memory, rng, context_id, best_action)
        for context_id, queries in queries_by_context.items():
            for query in queries:
                # No router means the previously unseen wording is a new key.
                exact_action = memory.recommend(query, list(ACTIONS))
                routed_action = memory.recommend(routes[query].context_id, list(ACTIONS))
                oracle_action = memory.recommend(context_id, list(ACTIONS))
                exact_correct += exact_action == best_actions[context_id]
                routed_correct += routed_action == best_actions[context_id]
                oracle_correct += oracle_action == best_actions[context_id]
                if query in automatically_accepted:
                    selective_correct += routed_action == best_actions[context_id]
                    selective_total += 1
                total += 1
        event_counts.append(memory.event_count)

    return {
        "config": {
            "seeds": seeds,
            "embed_backend": "bge",
            "probe_set": probe_set,
            "contexts": len(CONTEXTS),
            "held_out_queries_per_context": len(next(iter(queries_by_context.values()))),
            "feedback_per_context_action": 8,
            "feedback": "verified simulated outcomes before held-out routing queries",
            "min_margin": min_margin,
        },
        "routing": {
            "correct": route_correct,
            "total": route_total,
            "accuracy": route_correct / route_total,
            "selective": _selective_routing(routes, expected_contexts, min_margin),
            "routes": {
                query: {
                    "expected_context_id": expected_contexts[query],
                    "context_id": route.context_id,
                    "similarity": route.similarity,
                    "runner_up_context_id": route.runner_up_context_id,
                    "runner_up_similarity": route.runner_up_similarity,
                    "margin": route.margin,
                }
                for query, route in routes.items()
            },
        },
        "summary": {
            "exact_context_key": exact_correct / total,
            "semantic_rag_route_plus_outcome_memory": routed_correct / total,
            "oracle_context_route_plus_outcome_memory": oracle_correct / total,
            "selective_semantic_route_plus_outcome_memory": {
                "automatic_action_accuracy": selective_correct / selective_total,
                "automatic_correct_share_of_all_queries": selective_correct / total,
                "abstentions_require_fallback": route_total - len(automatically_accepted),
            },
        },
        "resources": {
            "mean_outcome_events": sum(event_counts) / len(event_counts),
            "router_prototypes": len(prototypes),
            "held_out_queries": route_total,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.02,
        help="Do not automatically route examples whose top-two similarity gap is smaller.",
    )
    parser.add_argument(
        "--probe-set",
        choices=("calibration", "evaluation"),
        default="calibration",
        help="Use calibration probes or the post-threshold hand-authored evaluation probes.",
    )
    parser.add_argument(
        "--embed-model",
        default=("/home/omnichat/.cache/modelscope/models/"
                 "BAAI--bge-small-zh-v1.5/snapshots/master"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "semantic-transfer-bge.json",
    )
    args = parser.parse_args()
    result = run(
        seeds=args.seeds,
        embed_model=args.embed_model,
        min_margin=args.min_margin,
        probe_set=args.probe_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"routing": result["routing"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
