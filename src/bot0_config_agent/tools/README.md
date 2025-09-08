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