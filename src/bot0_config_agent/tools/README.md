
```json
{
  "list_project_files": {
    "name": "list_project_files",                // matches ToolCall.tool
    "description": "Recursively scans a project directory for files, supporting directory exclusion and file type inclusion filters. Returns files as a list of string paths.",
    "import_path": "tools.list_project_files.list_project_files",
    "parameters": {
      "type": "object",
      "properties": {
        "root": {
          "type": "string",
          "description": "The root directory to scan. Defaults to the project root if not provided."
        },
        "exclude": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of directory names or substrings to exclude."
        },
        "include": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of file extensions to include."
        }
      },
      "required": []
    },
    "output_model": "ListProjectFilesResult"     // ties to the Pydantic output type
  },

  "make_virtualenv": {
    "name": "make_virtualenv",
    "description": "Creates a Python virtual environment at a path.",
    "import_path": "tools.make_virtualenv.make_virtualenv",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "Where to create the environment."
        }
      },
      "required": ["path"]
    },
    "output_model": "MakeVirtualEnvResult"
  }
}
```
The tool_registry.json follows strictly key: value format for name, description, import_path, and parameters. 