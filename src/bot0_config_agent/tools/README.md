# How Does the Tool Registry Work:
## _manifest.yaml

### Purpose of `_manifest.yaml`

It’s a **catalog file** that tells your loader (`ToolRegistry`) how to gather all tool JSON schemas, and optionally how to treat them when building prompts or executing.

It separates:

* **where tools live** (the `include` lists)
* **default settings for all tools** (the `defaults`)
* **special exceptions** (the `overrides`)

This keeps your per-tool JSON files clean — they only describe schema (parameters, models, import path). The manifest handles planner- or project-specific tweaks.

---

### Sections

#### 1. `version`

```yaml
version: 1
```

A version number so you can evolve the manifest format later without breaking loaders.

---

#### 2. `load`

```yaml
load:
  - group: core
    include:
      - list_project_files.json
      - read_files.json
      - select_files.json
```

* **Groups**: logical categories (core, filesystem, config, env, etc.).
* **include**: which schema files to load for that group.
* **order respected**: helpful if you want a stable prompt order.

---

#### 3. `defaults`

```yaml
defaults:
  enabled: true
  hidden_from_planner: false
  tags: []
```

Applied to *all tools* unless explicitly overridden.

* `enabled`: whether the tool is active.
* `hidden_from_planner`: whether to expose in planning prompts.
* `tags`: freeform labels for later filtering.

---

#### 4. `overrides`

```yaml
overrides:
  llm_response_async.json:
    hidden_from_planner: true
    tags: [llm, fallback]

  select_files.json:
    prompt_alias: "select_files(paths: [str], include_ext?, exclude_ext?, ...)"
```

This is the exception map:

* You match a tool schema file (`llm_response_async.json`).
* Then specify properties that should replace or extend the defaults.
* Common uses:

  * Hide fallback tools from the planner (`hidden_from_planner: true`)
  * Add tags for retrieval or analytics (`tags: [config, secrets]`)
  * Provide a compact `prompt_alias` signature to keep prompts shorter.

---

#### 5. Optional: `planner`

```yaml
planner:
  max_tools_exposed: 12
  prefer_groups: ["core", "filesystem", "config"]
```

Hints for your planner:

* **`max_tools_exposed`**: cap how many tools to inject into a prompt.
* **`prefer_groups`**: if you use retrieval or filtering, pick from these groups first.

---

### Why `overrides` instead of editing JSON files?

* Keeps tool schemas **pure and reusable** — they only describe the contract.
* Overrides let you control planner behavior without polluting schema with prompt-only metadata.
* You can flip a flag in `_manifest.yaml` to hide or re-enable a tool without editing the schema file.
* Easier for experimentation: one manifest edit can change planner behavior for multiple tools.

---


# Save to File Tool – High-Level Plan

The goal is to enable the agent to **persist intermediate or final artifacts** (schemas, summaries, code outputs) into files, so they can be reused across sessions.

---

## 1. Motivation

* Generated schemas and summaries should not vanish after a single run.
* Having a **consistent naming convention** makes it easy for LLMs to reload and analyze them later.
* Supports **flat filenames** with embedded path or a **project/…/file.json** hierarchy.

---

## 2. Core Components

1. **Tool Schema (`tool_registry.json`)**

   * Define a new tool: `save_to_file`.
   * Parameters:

     * `content` (string or object) – what to save.
     * `path` (string) – logical/relative path for storage (e.g., `project/schemas/...json`).
     * `format` (enum: json, yaml, txt) – how to serialize.

2. **Tool Implementation**

   * Reads params, ensures directory exists.
   * Serializes `content` in the requested format.
   * Writes file safely (atomic write or temp + rename).
   * Returns success metadata (saved path, size, hash).

3. **File Naming Strategy**

   * **Option A (hierarchical):**

     ```
     project/schemas/bot0_config_agent.agent.planner.py.json
     ```

     Mirrors module path for clarity.
   * **Option B (flat with path embedded in filename):**

     ```
     project.schemas.bot0_config_agent.agent.planner.py.json
     ```

     Avoids deep directories, keeps everything under `schemas/`.
   * **Optional Manifest:**
     Maintain a JSON manifest mapping logical names → actual file paths, with hashes for integrity.

4. **Integration with Planner/Executor**

   * Add tool entry into `tool_registry.json`.
   * Register any required transformation (e.g., convert ToolResult → `content` string).
   * FSM/Planner can now route steps like `summarize_files → save_to_file`.

5. **Optional Enhancements**

   * **Hashing:** Compute SHA256 per file for deduplication.
   * **Manifest:** Keep `manifest.json` with `{ "file": "...", "hash": "...", "created": ... }`.
   * **Compression:** Allow `.gz` option for very large outputs.

---

## 3. Workflow Example

1. **Planner decides:**

   ```
   [
     { "tool": "summarize_files", "params": { "files": [...] } },
     { "tool": "save_to_file", "params": { 
         "content": "<step_0.summary>",
         "path": "project/summaries/cli.py.json",
         "format": "json"
     }}
   ]
   ```

2. **Executor runs:**

   * `summarize_files` → produces summary.
   * `save_to_file` → writes `cli.py.json`.

3. **Result:**

   * Summary persisted under predictable name.
   * Next run can `read_files` or `summarize_config` against it.

---

## 4. Deliverables to Build

* [ ] **Implementation module** in `tools/tool_scripts/save_to_file.py`.
* [ ] **Serialization helpers** (JSON/YAML/text).
* [ ] **Tests**: verify round-trip save & reload.
* [ ] **Planner integration**: add transformations if needed.
* [ ] **Manifest management** (optional, phase 2).

---

