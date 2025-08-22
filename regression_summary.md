# Regression Summary


## GPT — 3/4 passing


| Test | Result | Time (s) | Note |
|---|---|---:|---|
| YAML summary | ✅ WORKING | 14.43 | matched /"status"\s*:\s*"success"/ |
| List ./agent excl | ✅ WORKING | 2.71 | matched /"status"\s*:\s*"success"/ |
| Locate planner.py | ✅ WORKING | 3.21 | matched /"status"\s*:\s*"success"/ |
| Count files in agent | ❌ FAIL | 2.61 | rc=0; no success patterns matched; stderr_len=8356 |

## GEMINI — 1/2 passing


| Test | Result | Time (s) | Note |
|---|---|---:|---|
| Count files in agent | ❌ FAIL | 4.51 | rc=0; no success patterns matched; stderr_len=8279 |
| List ./agent excl | ✅ WORKING | 4.42 | matched /"status"\s*:\s*"success"/ |

## LLAMA2 — 0/1 passing


| Test | Result | Time (s) | Note |
|---|---|---:|---|
| Count files in tools | ❌ FAIL | 28.61 | rc=0; no success patterns matched; stderr_len=23461 |

## LLAMA3 — 1/2 passing


| Test | Result | Time (s) | Note |
|---|---|---:|---|
| Count files in agent | ❌ FAIL | 26.29 | rc=0; no success patterns matched; stderr_len=8995 |
| YAML summary | ✅ WORKING | 47.41 | matched /"status"\s*:\s*"success"/ |

## LIQUID — 3/4 passing


| Test | Result | Time (s) | Note |
|---|---|---:|---|
| List ./agent (miscounts ~10) | ✅ WORKING | 11.56 | matched /"status"\s*:\s*"success"/ |
| Count files in agent | ✅ WORKING | 12.45 | matched /"status"\s*:\s*"success"/ |
| Count files in project | ✅ WORKING | 8.77 | matched /"status"\s*:\s*"success"/ |
| List files in prompts | ❌ FAIL | 12.89 | rc=0; no success patterns matched; stderr_len=11580 |

**TOTAL:** 8/13 passing
