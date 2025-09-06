
---

## Tool Output → Input Handoff

The agent uses a **structured transformation layer** to pass results between tools in a multi-step plan without hardcoding parameter wiring.

### How it works

1. **Tool Execution**

   * Each step is executed by the `ToolChainExecutor` under FSM control.
   * The output is normalized into a standard envelope:

     ```json
     {
       "status": "success",
       "result": ...,
       "message": ""
     }
     ```

2. **Transformation Lookup**

   * After a tool finishes, the executor checks `tool_transformation.json` for a matching `{source_tool, target_tool}` rule.
   * If `transform_fn` is:

     * **`null`** → direct self-validation: output payload is fed into the next tool’s Pydantic input model as-is.
     * **A function path** → that function transforms the payload into the correct input shape.

3. **Model Validation**

   * The transformed payload is validated against the **target tool’s input model** (`pydantic.BaseModel`).
   * On success, it’s stored in `_next_params[next_idx]` for injection into the next step.
   * On failure, a warning is logged and the unmodified output is left in `context`.

4. **Next Step Injection**

   * When the next tool runs, its parameters are:

     * Taken from the plan.
     * Overridden by any auto-converted `_next_params` from the prior step.

### Advantages

* **No hardcoded field names** between tools.
* **Self-healing**: if a transformation fails, the original output is still available.
* **Pydantic enforcement** ensures the next tool receives exactly the fields it expects.
* **Extensible**: add new source→target transformations just by editing `tool_transformation.json` and implementing the function.

---

If you want, I can also give you a **diagram** showing this data flow from Tool A → transform → Tool B for the README. That would make the process much easier to visualize.
