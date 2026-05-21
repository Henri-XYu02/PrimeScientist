## FIRE-Bench Self-Improvement Search Pipeline

```mermaid
flowchart TD
    START([run_search.py]) --> INIT

    subgraph INIT["① Initialization"]
        direction LR
        CHK{Tree on disk?} -- "No / --force_init" --> REFLINIT["Reflector: init_task\n─────────────────\nReads: instruction.md\n       cross-task insights/\nWrites: root/skill.md\n        root/{task}_insight.md"]
        CHK -- Yes --> RESUME[Resume from disk\nload_tree]
        REFLINIT --> RESUME
    end

    RESUME --> LOOP

    subgraph LOOP["② Main Loop  ( while evals_done < budget  AND  tokens/time not exceeded )"]
        direction TB

        COLLECT["collect_unrun_leaves(root)\n────────────────────────────\nDFS over tree, PUCT-sorted\nRespects: max_depth · regression_threshold\n          subtree_exhausted · proposals cap"]

        COLLECT --> GATE{Unrun\nleaves?}

        GATE -- "Yes → fill parallel slots" --> PEVAL

        subgraph PEVAL["③ Parallel Evaluation  ( ThreadPoolExecutor )"]
            direction LR
            SN["setup_node\nstage skill.md\n→ sandbox"] --> RA
            RA["run_agent\nclaud / codex\nsubprocess\n(cwd = sandbox)"] --> EV
            EV["evaluate\ndocker build cached\ndocker run tests\nparse score"] --> MA
            MA["_move_artifacts\nsandbox → node/sandbox\nlog    → node/log.log"]
        end

        PEVAL --> WRITE["write_packet  ·  backpropagate\nappend results.tsv  ·  append costs.tsv\n────────────────────────────────────────\n(serialized with threading.Lock)"]
        WRITE --> BUDCHK{Budget\nexceeded?}
        BUDCHK -- No --> COLLECT
        BUDCHK -- "Yes (tokens / time / evals)" --> DONE

        GATE -- No --> SEL["puct_select_leaf\n───────────────────\nFinds best run node where:\n  len(children) < --proposals  AND\n  all child subtrees exhausted\n  (adaptive branching gate)"]

        SEL --> CAPCHK{Can add\nmore children?}
        CAPCHK -- "No → at cap or\ndepth/regression stop" --> DONE

        CAPCHK -- Yes --> REFL

        subgraph REFL["④ Reflection  ( sequential )"]
            direction LR
            RAGENT["Reflector agent\nClaude / Codex\n─────────────────────\nReads from tree dir:\n  instruction.md\n  node/log.log\n  node/sandbox/\n  node/packet.json\n  prior diff.py files\n  sibling stats.json"] --> PROP
            PROP["Writes 1 … N proposals\n(agent decides count, cap enforced)\n────────────────────────────────\nEach proposal dir contains:\n  diff.py  → runs → skill.md\n  prior.json  {estimate, rationale}\n\nProposals must be CONTROVERSIALLY\nDIFFERENT hypotheses — not param tweaks"]
        end

        REFL --> COLLECT
    end

    DONE([Search complete])

    subgraph TREE["Search Tree on Disk  search_tree/{task}/"]
        direction TB
        ROOT["root/\n  skill.md\n  packet.json  ← score, cost, conclusion\n  stats.json   ← Q, visits, P\n  log.log\n  sandbox/"]
        ROOT --> C0["children/proposal_0/\n  diff.py  skill.md\n  prior.json  packet.json\n  stats.json  log.log  sandbox/\n  children/ …"]
        ROOT --> C1["children/proposal_1/\n  …"]
        C0 --> GC["children/proposal_0/\n  …  (up to max_depth)"]
    end

    subgraph LEDGERS["Ledgers  (per task)"]
        direction LR
        RES["results.tsv\nnode · score · ran_at"]
        COST["costs.tsv\nrole · model · cost_usd\ninput_tokens · output_tokens\ncache_read · cache_write · node"]
    end

    subgraph BENCH["Benchmark Backends"]
        direction TB
        AL["AutoLab\nenvironment/ → sandbox\ndocker eval → reward.json\n(reward 0–1)"]
        TB2["Terminal-Bench\ndocker cp /app → sandbox\nagent edits files\ndocker run pytest → pass ratio"]
        FB["FIRE-Bench\ndata/ → sandbox\nRAGChecker F1 vs ground truth"]
    end

    WRITE --> LEDGERS
    REFL --> COST
    PEVAL -. uses .-> BENCH
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Only add proposals after full subtree exhaustion | Never widen before the current depth is fully explored — avoids premature branching |
| Reflector decides proposal count (1 … cap) | LLM judges diversity better than a fixed formula; proposals must be *controversially different* |
| Parallel eval, sequential reflection | Agent runs are independent (separate sandboxes); reflection needs prior results |
| `costs.tsv` tracks both agent + reflector | Single file gives total token spend per task across all components |
| `--max_tokens` / `--max_time_sec` | Enables fair comparison across methods under the same resource budget |

---

## Experimental Comparison Framework

```mermaid
flowchart LR
    BUDGET(["Token / Time Budget\n(same for all methods)"])

    BUDGET --> M1
    BUDGET --> M2
    BUDGET --> M3

    subgraph M1["① FIRE-Bench Tree Search  (this system)"]
        direction TB
        M1A["PUCT-guided tree\n--proposals N  --parallel K\n--puct_c  --max_depth"]
        M1B["Reflector proposes\n1…N controversially different\nhypotheses per branch"]
        M1C["Parallel eval fills\nall unrun leaves\nbefore widening"]
        M1A --> M1B --> M1C
    end

    subgraph M2["② Linear Search  (ablation: 1 proposal)"]
        direction TB
        M2A["Single-child chain\n--proposals 1\n--parallel 1"]
        M2B["Reflector proposes\nexactly 1 variant\n(greedy best path)"]
        M2C["Depth-first:\ndeepens before widening\n(no branching)"]
        M2A --> M2B --> M2C
    end

    subgraph M3["③ Vanilla AutoResearch  (baseline)"]
        direction TB
        M3A["No tree — flat loop\nsequential runs\none plan at a time"]
        M3B["Plan improves via\nfull-context reflection\n(no PUCT, no priors)"]
        M3C["Budget tracked\nby eval count only\n(no token/time gate)"]
        M3A --> M3B --> M3C
    end

    M1 --> EVAL
    M2 --> EVAL
    M3 --> EVAL

    subgraph EVAL["Evaluation  (same benchmark, same task)"]
        direction LR
        SC["Best score achieved\nwithin budget"]
        EFF["Score vs tokens\n(efficiency curve)"]
        VAR["Score variance\nacross runs"]
        SC --- EFF --- VAR
    end
```

---

## Token Budget vs Score Curve  (expected shape)

```mermaid
xychart-beta
    title "Score vs Token Budget (schematic)"
    x-axis ["0", "50k", "100k", "200k", "400k", "800k"]
    y-axis "Best Score (F1 / reward)" 0 --> 1
    line [0.0, 0.35, 0.52, 0.68, 0.74, 0.76]
    line [0.0, 0.33, 0.48, 0.60, 0.62, 0.62]
    line [0.0, 0.28, 0.40, 0.50, 0.53, 0.54]
```

> **Legend (top to bottom):** Tree Search (--proposals 2) · Linear Search (--proposals 1) · Vanilla AutoResearch
>
> Tree search should dominate once the tree has enough budget to branch — both methods are similar at very low budgets where only 1–2 evals are possible.

---

## Search Tree Depth vs Score  (observed on autolab tasks)

```mermaid
xychart-beta
    title "Mean Best Score by Tree Depth"
    x-axis "Tree Depth" [0, 1, 2, 3, 4, 5]
    y-axis "Mean Score (across tasks)" 0 --> 1
    bar  [0.30, 0.45, 0.58, 0.71, 0.68, 0.60]
    line [0.30, 0.45, 0.58, 0.71, 0.68, 0.60]
```

> Peak performance at depth 3; decline at depth 4–5 suggests overfitting of skill.md to node-specific noise.
> Motivates `--max_depth 4` and lower `--puct_c 1.0` (exploit more, explore less).
