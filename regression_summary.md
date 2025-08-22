# Regression Summary

## GPT (API) — 3/4 passing

| Test | Result | Time (s) | Note |
|---|---|---:|---|
| YAML summary | ✅ WORKING | 12.87 | matched /"status"\s*:\s*"success"/ |
| List ./agent excl | ✅ WORKING | 2.96 | matched /"status"\s*:\s*"success"/ |
| Locate planner.py | ✅ WORKING | 4.15 | matched /"status"\s*:\s*"success"/ |
| Count files in agent | ❌ FAIL | 2.41 | rc=0; no success patterns matched; stderr_len=8356 |

## GEMINI (API) — 1/2 passing

| Test | Result | Time (s) | Note |
|---|---|---:|---|
| Count files in agent | ❌ FAIL | 4.97 | rc=0; no success patterns matched; stderr_len=8279 |
| List ./agent excl | ✅ WORKING | 4.32 | matched /"status"\s*:\s*"success"/ |

## LLAMA_2_7B_GPTQ (local) — 0/1 passing

| Test | Result | Time (s) | Note |
|---|---|---:|---|
| Count files in tools | ❌ FAIL | 20.42 | rc=0; no success patterns matched; stderr_len=23461 |

## LLAMA_3_2_3B_GPTQ (local) — 1/2 passing

| Test | Result | Time (s) | Note |
|---|---|---:|---|
| Count files in agent | ❌ FAIL | 12.03 | rc=0; no success patterns matched; stderr_len=8994 |
| YAML summary | ✅ WORKING | 44.79 | matched /"status"\s*:\s*"success"/ |

## LIQUID lfm2_1_2b (local) — 4/4 passing

| Test | Result | Time (s) | Note |
|---|---|---:|---|
| List ./agent (miscounts ~10) | ✅ WORKING | 7.48 | matched /"status"\s*:\s*"success"/ |
| Count files in agent | ✅ WORKING | 9.73 | matched /"status"\s*:\s*"success"/ |
| Count files in project | ✅ WORKING | 7.99 | matched /"status"\s*:\s*"success"/ |
| List files in prompts | ✅ WORKING | 8.93 | matched /"status"\s*:\s*"success"/ |

**TOTAL:** 9/13 passing
