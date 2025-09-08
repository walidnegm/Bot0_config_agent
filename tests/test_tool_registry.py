import json
import sys
from pathlib import Path

import pytest

# Adjust imports if your paths differ; with conftest.py these should work.
from bot0_config_agent.tools.configs.tool_registry import ToolRegistry
from bot0_config_agent.tools.configs.tool_registry_loader import load_tool_registry
from bot0_config_agent.tools.configs.tool_models import ToolSpec, ToolOutput
from bot0_config_agent.agent_models.step_status import StepStatus


def _write_pkg(tmp_path: Path, pkg_name: str = "tmp_pkg"):
    """
    Create a temporary, importable Python package with:
      - tmp_pkg/__init__.py
      - tmp_pkg/mod.py       (has function 'do_it')
      - tmp_pkg/models.py    (has pydantic model 'FilesInput')
      - tmp_pkg/transform.py (has function 'to_files_input')
    """
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(
        "def do_it(**kwargs):\n"
        "    return {'status': 'SUCCESS', 'result': kwargs, 'message': ''}\n"
    )
    (pkg / "models.py").write_text(
        "from pydantic import BaseModel\n"
        "class FilesInput(BaseModel):\n"
        "    result: dict\n"
    )
    (pkg / "transform.py").write_text(
        "def to_files_input(payload):\n"
        "    if isinstance(payload, dict):\n"
        "        return payload\n"
        "    return {'result': {'files': payload}}\n"
    )

    # Make the tmp package importable
    sys.path.insert(0, str(tmp_path))
    return pkg


def _tool_json(
    name: str,
    import_path: str,
    *,
    params_properties: dict | None = None,
    input_model: str | None = None,
):
    return {
        "name": name,
        "description": f"desc for {name}",
        "import_path": import_path,
        "parameters": {
            "type": "object",
            "properties": params_properties or {"foo": {"type": "string"}},
            "required": [],
        },
        "input_model": input_model,
        "output_model": None,
    }


def test_directory_registry_basic(tmp_path: Path):
    pkg = _write_pkg(tmp_path)

    # Create dir registry with one tool
    regdir = tmp_path / "registry"
    regdir.mkdir()

    tool_path = regdir / "echo.tool.json"
    tool_spec = _tool_json(
        name="echo",
        import_path=f"{pkg.name}.mod.do_it",
        params_properties={"message": {"type": "string"}},
    )
    tool_path.write_text(json.dumps(tool_spec, indent=2))

    # Minimal manifest
    (regdir / "_manifest.yaml").write_text("include:\n  - '.'\n")

    # Load registry via ToolRegistry (dir mode)
    tr = ToolRegistry(
        registry_path=regdir, transformation_path=None, import_sanity_check=True
    )
    tools = tr.get_all()
    assert "echo" in tools
    assert isinstance(tools["echo"], ToolSpec)

    # get_function
    fn = tr.get_function("echo")
    assert callable(fn)

    # get_param_keys
    keys = tr.get_param_keys("echo")
    assert keys == {"message"}


def test_manifest_overrides(tmp_path: Path):
    pkg = _write_pkg(tmp_path)

    regdir = tmp_path / "registry2"
    regdir.mkdir()

    tool_rel = Path("sub/alpha.tool.json")
    (regdir / "sub").mkdir()
    (regdir / tool_rel).write_text(
        json.dumps(
            _tool_json(
                name="alpha",
                import_path=f"{pkg.name}.mod.do_it",
                params_properties={"p": {"type": "integer"}},
            ),
            indent=2,
        )
    )

    # Override description and parameters via manifest
    (regdir / "_manifest.yaml").write_text(
        "include:\n"
        "  - '.'\n"
        "overrides:\n"
        f"  '{tool_rel.as_posix()}':\n"
        "    description: 'OVERRIDDEN'\n"
        "    parameters:\n"
        "      type: object\n"
        "      properties:\n"
        "        q:\n"
        "          type: string\n"
        "      required: []\n"
    )

    tr = ToolRegistry(registry_path=regdir, transformation_path=None)
    spec = tr.get_all()["alpha"]
    assert spec.description == "OVERRIDDEN"
    assert "q" in spec.parameters.get("properties", {})


def test_legacy_file_loading(tmp_path: Path):
    pkg = _write_pkg(tmp_path)

    # Legacy single JSON file mapping name -> spec
    legacy = tmp_path / "tool_registry.json"
    legacy.write_text(
        json.dumps(
            {
                "legacy_tool": _tool_json(
                    name="legacy_tool",
                    import_path=f"{pkg.name}.mod.do_it",
                    params_properties={"x": {"type": "string"}},
                )
            },
            indent=2,
        )
    )

    tr = ToolRegistry(
        registry_path=legacy, transformation_path=None, import_sanity_check=True
    )
    tools = tr.get_all()
    assert "legacy_tool" in tools
    assert tr.get_param_keys("legacy_tool") == {"x"}


def test_duplicate_tool_name_raises(tmp_path: Path):
    pkg = _write_pkg(tmp_path)
    regdir = tmp_path / "registry_dup"
    regdir.mkdir()

    # Two files with same tool.name = "dup"
    (regdir / "a.tool.json").write_text(
        json.dumps(
            _tool_json(name="dup", import_path=f"{pkg.name}.mod.do_it"), indent=2
        )
    )
    (regdir / "b.tool.json").write_text(
        json.dumps(
            _tool_json(name="dup", import_path=f"{pkg.name}.mod.do_it"), indent=2
        )
    )

    (regdir / "_manifest.yaml").write_text("include:\n  - '.'\n")

    with pytest.raises(ValueError):
        _ = ToolRegistry(registry_path=regdir, transformation_path=None)


def test_import_sanity_check_failure(tmp_path: Path):
    # import path points to missing function
    pkg = _write_pkg(tmp_path)
    regdir = tmp_path / "registry_bad"
    regdir.mkdir()
    (regdir / "_manifest.yaml").write_text("include:\n  - '.'\n")

    (regdir / "bad.tool.json").write_text(
        json.dumps(
            _tool_json(
                name="bad", import_path=f"{pkg.name}.mod.NOPE"  # does not exist
            ),
            indent=2,
        )
    )

    with pytest.raises(ImportError):
        _ = ToolRegistry(
            registry_path=regdir, transformation_path=None, import_sanity_check=True
        )


def test_match_and_convert_with_transform(tmp_path: Path):
    pkg = _write_pkg(tmp_path)

    # Registry dir with two tools: producer -> consumer
    regdir = tmp_path / "registry_tx"
    regdir.mkdir()
    (regdir / "_manifest.yaml").write_text("include:\n  - '.'\n")

    # Producer tool
    (regdir / "producer.tool.json").write_text(
        json.dumps(
            _tool_json(
                name="producer",
                import_path=f"{pkg.name}.mod.do_it",
                params_properties={"seed": {"type": "integer"}},
            ),
            indent=2,
        )
    )

    # Consumer expects a Pydantic model at tmp_pkg.models.FilesInput
    consumer_spec = _tool_json(
        name="consumer",
        import_path=f"{pkg.name}.mod.do_it",
        params_properties={"result": {"type": "object"}},
        input_model=f"{pkg.name}.models.FilesInput",
    )
    (regdir / "consumer.tool.json").write_text(json.dumps(consumer_spec, indent=2))

    # Transform map file
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            [
                {
                    "source_tool": "producer",
                    "target_tool": "consumer",
                    "transform_fn": f"{pkg.name}.transform.to_files_input",
                }
            ],
            indent=2,
        )
    )

    tr = ToolRegistry(
        registry_path=regdir, transformation_path=transforms, import_sanity_check=True
    )

    # Simulate producer output
    out = ToolOutput(status=StepStatus.SUCCESS, message="", result=["a.txt", "b.txt"])

    # Convert to consumer input via (source,target) mapping
    model = tr.match_and_convert_output(
        output=out, target_tool="consumer", source_tool="producer"
    )
    assert hasattr(model, "result")
    assert isinstance(model.result, dict)
    assert model.result["files"] == ["a.txt", "b.txt"]


def test_get_function_returns_callable(tmp_path: Path):
    pkg = _write_pkg(tmp_path)
    regdir = tmp_path / "registry_fn"
    regdir.mkdir()
    (regdir / "_manifest.yaml").write_text("include:\n  - '.'\n")
    (regdir / "t.tool.json").write_text(
        json.dumps(_tool_json(name="t", import_path=f"{pkg.name}.mod.do_it"), indent=2)
    )

    tr = ToolRegistry(registry_path=regdir, transformation_path=None)
    fn = tr.get_function("t")
    assert callable(fn)
