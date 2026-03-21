import json
import re
from typing import Any, Dict, List, Optional, Set, Type, Union
import interegular
from interegular.fsm import FSM
from pydantic import BaseModel

WS = r"[ ]*"
STRING_REGEX = r'"([^"\\\x00-\x1f\n\r\t]{0,24})"'
INTEGER_REGEX = r"-?(0|[1-9][0-9]{0,5})"
NUMBER_REGEX = r"-?(0|[1-9][0-9]{0,5})(\.[0-9]{1,3})?"
BOOLEAN_REGEX = r"(true|false)"
NULL_REGEX = r"null"


def escape_string_for_regex(s: str) -> str:
    return re.escape(s)


class JSONSchemaToRegexCompiler:
    def __init__(self, whitespace: str = WS):
        self.ws = whitespace

    def compile(self, schema: Union[Dict[str, Any], Type[BaseModel]]) -> str:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

        defs = schema_dict.get("$defs", schema_dict.get("definitions", {}))
        return self._compile_node(schema_dict, defs)

    def _compile_node(self, node: Dict[str, Any], defs: Dict[str, Any]) -> str:
        if "$ref" in node:
            ref_path = node["$ref"]
            ref_name = ref_path.split("/")[-1]
            if ref_name in defs:
                return self._compile_node(defs[ref_name], defs)
            raise ValueError(f"Unresolved reference: {ref_path}")

        if "anyOf" in node:
            options = [self._compile_node(opt, defs) for opt in node["anyOf"]]
            return f"({'|'.join(options)})"
        if "oneOf" in node:
            options = [self._compile_node(opt, defs) for opt in node["oneOf"]]
            return f"({'|'.join(options)})"

        if "enum" in node:
            enum_patterns = []
            for val in node["enum"]:
                if isinstance(val, str):
                    enum_patterns.append(f'"{escape_string_for_regex(val)}"')
                elif isinstance(val, bool):
                    enum_patterns.append("true" if val else "false")
                elif isinstance(val, (int, float)):
                    enum_patterns.append(str(val))
                elif val is None:
                    enum_patterns.append("null")
                else:
                    enum_patterns.append(escape_string_for_regex(json.dumps(val)))
            return f"({'|'.join(enum_patterns)})"

        node_type = node.get("type")

        if isinstance(node_type, list):
            type_options = []
            for t in node_type:
                sub_node = {**node, "type": t}
                type_options.append(self._compile_node(sub_node, defs))
            return f"({'|'.join(type_options)})"

        if node_type == "string":
            if "pattern" in node:
                return f'"{node["pattern"]}"'
            return STRING_REGEX

        elif node_type == "integer":
            return INTEGER_REGEX

        elif node_type == "number":
            return NUMBER_REGEX

        elif node_type == "boolean":
            return BOOLEAN_REGEX

        elif node_type == "null":
            return NULL_REGEX

        elif node_type == "array":
            return self._compile_array(node, defs)

        elif node_type == "object" or "properties" in node:
            return self._compile_object(node, defs)

        return f"({STRING_REGEX}|{NUMBER_REGEX}|{BOOLEAN_REGEX}|{NULL_REGEX})"

    def _compile_array(self, node: Dict[str, Any], defs: Dict[str, Any]) -> str:
        items_node = node.get("items", {})
        if not items_node:
            item_regex = f"({STRING_REGEX}|{NUMBER_REGEX}|{BOOLEAN_REGEX}|{NULL_REGEX})"
        else:
            item_regex = self._compile_node(items_node, defs)

        min_items = node.get("minItems", 0)

        if min_items == 0:
            elements = f"({item_regex}({self.ws},{self.ws}{item_regex}){{0,2}})?"
        else:
            additional_items = f"({self.ws},{self.ws}{item_regex})" * (min_items - 1)
            remaining = f"({self.ws},{self.ws}{item_regex}){{0,2}}"
            elements = f"{item_regex}{additional_items}{remaining}"

        return rf"\[{self.ws}{elements}{self.ws}\]"

    def _compile_object(self, node: Dict[str, Any], defs: Dict[str, Any]) -> str:
        properties = node.get("properties", {})
        required_props = set(node.get("required", []))

        if not properties:
            return rf"\{{{self.ws}\}}"

        prop_patterns: List[str] = []
        for prop_name, prop_schema in properties.items():
            escaped_key = escape_string_for_regex(prop_name)
            value_pattern = self._compile_node(prop_schema, defs)
            prop_patterns.append(rf'"{escaped_key}"{self.ws}:{self.ws}{value_pattern}')

        if required_props == set(properties.keys()):
            joined_props = f"{self.ws},{self.ws}".join(prop_patterns)
            return rf"\{{{self.ws}{joined_props}{self.ws}\}}"
        else:
            joined_props = f"{self.ws},{self.ws}".join(prop_patterns)
            return rf"\{{{self.ws}({joined_props})?{self.ws}\}}"


class FSMCompiler:
    def __init__(self, whitespace: str = WS):
        self.schema_compiler = JSONSchemaToRegexCompiler(whitespace=whitespace)

    def compile_schema_to_fsm(self, schema: Union[Dict[str, Any], Type[BaseModel]]) -> FSM:
        regex_pattern = self.schema_compiler.compile(schema)
        return self.compile_regex_to_fsm(regex_pattern)

    def compile_regex_to_fsm(self, regex_pattern: str) -> FSM:
        try:
            parsed_pattern = interegular.parse_pattern(regex_pattern)
            return parsed_pattern.to_fsm()
        except Exception as e:
            raise ValueError(f"Failed to compile pattern '{regex_pattern}': {e}") from e


compiler = FSMCompiler()
