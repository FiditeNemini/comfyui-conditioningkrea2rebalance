from __future__ import annotations

import ast
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ComfyUI-OmniNode")



class OmniSpec:

    def __init__(self) -> None:
        self.class_name: str = ""
        self.function: str = "execute"
        self.required: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self.optional: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self.hidden: Dict[str, str] = {}
        self.return_types: Tuple[Any, ...] = ()
        self.return_names: Tuple[str, ...] = ()
        self.category: str = "omni"
        self.description: str = ""
        self.raw_source: str = ""
        self.error: Optional[str] = None

    def input_types_dict(self) -> Dict[str, Dict[str, Any]]:

        return {
            "required": {k: v for k, v in self.required.items()},
            "optional": {k: v for k, v in self.optional.items()},
            "hidden": dict(self.hidden),
        }

    def to_json(self) -> Dict[str, Any]:

        def _ser(spec_entry: Tuple[Any, Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
            t, opts = spec_entry
            if isinstance(t, (list, tuple)):
                t_ser = list(t)
            else:
                t_ser = t
            return (t_ser, dict(opts))

        return {
            "class_name": self.class_name,
            "function": self.function,
            "required": {k: _ser(v) for k, v in self.required.items()},
            "optional": {k: _ser(v) for k, v in self.optional.items()},
            "hidden": dict(self.hidden),
            "return_types": [self._ser_type(t) for t in self.return_types],
            "return_names": list(self.return_names),
            "category": self.category,
            "error": self.error,
        }

    @staticmethod
    def _ser_type(t: Any) -> Any:
        if isinstance(t, (list, tuple)):
            return list(t)
        return t



_KNOWN_TYPE_NAMES = {
    "IMAGE", "MASK", "LATENT", "CONDITIONING", "CLIP", "MODEL", "VAE",
    "CLIP_VISION", "CLIP_VISION_OUTPUT", "CONTROL_NET", "STYLE_MODEL",
    "UPSCALE_MODEL", "LORA", "SAM_MODEL", "BBOX", "BoundingBox",
    "STRING", "INT", "FLOAT", "BOOLEAN", "NUMBER", "AUDIO", "WAV",
    "COMBO", "Combo", "DICT", "*",
}


class _TypeResolver(ast.NodeTransformer):

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.Constant(value=node.id)


def _literal_eval_node(node: ast.AST) -> Any:

    transformed = _TypeResolver().visit(node)
    ast.fix_missing_locations(transformed)
    return ast.literal_eval(transformed)



def parse_omni_source(source: str) -> OmniSpec:

    spec = OmniSpec()
    spec.raw_source = source

    if not source or not source.strip():
        spec.error = "No code pasted."
        return spec

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        spec.error = f"Syntax error: {e}"
        return spec

    class_node = _find_first_node_class(tree)
    if class_node is not None:
        spec.class_name = class_node.name
        body = class_node.body
    else:
        spec.class_name = "OmniPasted"
        body = tree.body

    _extract_class_attrs(body, spec)

    if spec.return_types and not isinstance(spec.return_types, tuple):
        spec.return_types = (spec.return_types,)

    return spec


def _find_first_node_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            has_input_types = any(
                isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES"
                for n in node.body
            )
            if has_input_types:
                return node
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            has_return = any(
                isinstance(n, ast.Assign) and _assign_targets_to_names(n)
                for n in node.body
            )
            if has_return:
                return node
    return None


def _assign_targets_to_names(node: ast.Assign) -> List[str]:
    names: List[str] = []
    for t in node.targets:
        if isinstance(t, ast.Name):
            names.append(t.id)
    return names


def _extract_class_attrs(body: List[ast.stmt], spec: OmniSpec) -> None:
    for node in body:
        if isinstance(node, ast.Assign):
            names = _assign_targets_to_names(node)
            if not names:
                continue
            value = node.value
            for name in names:
                _handle_assignment(name, value, spec)
        elif isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES":
            spec.required, spec.optional, spec.hidden = _parse_input_types_func(node)


def _handle_assignment(name: str, value: ast.AST, spec: OmniSpec) -> None:
    if name == "FUNCTION":
        try:
            spec.function = ast.literal_eval(value)
        except Exception:
            pass
    elif name == "RETURN_TYPES":
        spec.return_types = _parse_return_types(value)
    elif name == "RETURN_NAMES":
        try:
            v = ast.literal_eval(value)
            if isinstance(v, (list, tuple)):
                spec.return_names = tuple(v)
        except Exception:
            pass
    elif name == "CATEGORY":
        try:
            spec.category = ast.literal_eval(value)
        except Exception:
            pass
    elif name == "DESCRIPTION":
        try:
            spec.description = ast.literal_eval(value)
        except Exception:
            pass


def _parse_return_types(value: ast.AST) -> Tuple[Any, ...]:

    if isinstance(value, ast.Tuple):
        out = []
        for el in value.elts:
            out.append(_parse_single_type(el))
        return tuple(out)
    return (_parse_single_type(value),)


def _parse_single_type(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.List, ast.Tuple)):
        try:
            return list(ast.literal_eval(node))
        except Exception:
            out = []
            for el in node.elts:
                if isinstance(el, ast.Constant):
                    out.append(el.value)
                elif isinstance(el, ast.Name):
                    out.append(el.id)
                else:
                    out.append("*")
            return out
    return "*"


def _parse_input_types_func(
    func_node: ast.FunctionDef,
) -> Tuple[Dict[str, Tuple[Any, Dict[str, Any]]], Dict[str, Tuple[Any, Dict[str, Any]]], Dict[str, str]]:

    required: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    optional: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    hidden: Dict[str, str] = {}

    ret_dict_node = _find_returned_dict(func_node)
    if ret_dict_node is None:
        return required, optional, hidden

    for key_node, value_node in zip(ret_dict_node.keys, ret_dict_node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        section = key_node.value
        if section == "hidden":
            try:
                hidden = ast.literal_eval(value_node)
            except Exception:
                hidden = {}
            continue
        if section not in ("required", "optional"):
            continue
        try:
            entries = _literal_eval_node(value_node)
        except Exception as e:
            logger.warning("Omni: failed to parse %s inputs: %s", section, e)
            continue
        if not isinstance(entries, dict):
            continue
        target = required if section == "required" else optional
        for in_name, in_val in entries.items():
            target[in_name] = _normalize_input_entry(in_val)

    return required, optional, hidden


def _find_returned_dict(func_node: ast.FunctionDef) -> Optional[ast.Dict]:

    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return node.value
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
            var_name = node.value.id
            for stmt in func_node.body:
                if (isinstance(stmt, ast.Assign)
                        and isinstance(stmt.value, ast.Dict)
                        and any(isinstance(t, ast.Name) and t.id == var_name for t in stmt.targets)):
                    return stmt.value
    return None


def _normalize_input_entry(entry: Any) -> Tuple[Any, Dict[str, Any]]:

    if isinstance(entry, str):
        return (entry, {})
    if isinstance(entry, tuple):
        if len(entry) == 2 and isinstance(entry[1], dict):
            t, opts = entry
            return (t, dict(opts))
        if len(entry) == 1:
            return (entry[0], {})
        return (entry[0] if entry else "*", {})
    if isinstance(entry, list):
        return (list(entry), {})
    if isinstance(entry, dict):
        t = entry.get("type", "*")
        opts = {k: v for k, v in entry.items() if k != "type"}
        return (t, opts)
    return ("*", {})



_SOURCE_CACHE: Dict[str, Tuple[type, OmniSpec]] = {}


def compile_omni_class(source: str) -> Tuple[Optional[type], OmniSpec]:

    spec = parse_omni_source(source)
    if spec.error:
        return None, spec
    if not spec.raw_source.strip():
        spec.error = "Empty source."
        return None, spec

    cached = _SOURCE_CACHE.get(source)
    if cached is not None:
        return cached[0], spec

    namespace: Dict[str, Any] = {}
    for name in _KNOWN_TYPE_NAMES:
        namespace[name] = name

    try:
        code = compile(spec.raw_source, "<omni_pasted>", "exec")
        exec(code, namespace)
    except Exception:
        spec.error = "Failed to compile/execute pasted source:\n" + traceback.format_exc()
        return None, spec

    cls = _find_class_in_namespace(namespace, spec.class_name)
    if cls is None:
        spec.error = f"Could not find class {spec.class_name!r} in pasted source."
        return None, spec

    _SOURCE_CACHE[source] = (cls, spec)
    return cls, spec


def _find_class_in_namespace(namespace: Dict[str, Any], name: str) -> Optional[type]:
    if name in namespace and isinstance(namespace[name], type):
        return namespace[name]
    for v in namespace.values():
        if isinstance(v, type) and hasattr(v, "INPUT_TYPES"):
            return v
    return None



class OmniNode:

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        base_required = {
            "code": ("STRING", {
                "default": "",
                "multiline": True,

            }),
        }
        base_optional = {
            "omni_spec": ("STRING", {
                "default": "",
                "multiline": False,
                "advanced": True,
            }),
        }
        base_hidden = {
            "prompt": "PROMPT",
            "unique_id": "UNIQUE_ID",
        }
        return {
            "required": base_required,
            "optional": base_optional,
            "hidden": base_hidden,
        }

    _WILDCARD_OUT = tuple("*" for _ in range(24))
    RETURN_TYPES = _WILDCARD_OUT
    RETURN_NAMES = tuple(f"output_{i + 1}" for i in range(24))
    FUNCTION = "execute"
    CATEGORY = "Rebalance-Pack/foundational"
    OUTPUT_IS_LIST = tuple(False for _ in range(24))

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> bool:
        return True

    def execute(self, code: str = "", **kwargs) -> Tuple[Any, ...]:
        kwargs.pop("omni_spec", None)

        prompt = kwargs.pop("prompt", None)
        unique_id = kwargs.pop("unique_id", None)
        if isinstance(prompt, list) and prompt:
            prompt = prompt[0]
        if isinstance(unique_id, list) and unique_id:
            unique_id = unique_id[0]

        cls, spec = compile_omni_class(code or "")
        if cls is None or spec.error:
            raise RuntimeError(f"OmniNode: {spec.error or 'unknown error'}")

        func_name = spec.function or "execute"
        func = getattr(cls(), func_name, None)
        if func is None:
            raise RuntimeError(
                f"OmniNode: pasted class {spec.class_name!r} has no function "
                f"{func_name!r}."
            )

        declared = set(spec.required.keys()) | set(spec.optional.keys())
        call_kwargs: Dict[str, Any] = {}

        for k, v in kwargs.items():
            if k in declared:
                call_kwargs[k] = v

        if prompt and unique_id is not None:
            node_inputs = None
            try:
                node_inputs = prompt[str(unique_id)]["inputs"]
            except (KeyError, TypeError):
                node_inputs = None
            if isinstance(node_inputs, dict):
                for name in declared:
                    if name in call_kwargs:
                        continue
                    if name not in node_inputs:
                        continue
                    val = node_inputs[name]
                    if isinstance(val, list) and len(val) == 2 and all(isinstance(x, (int, str)) for x in val):
                        continue
                    if isinstance(val, dict) and "__value__" in val:
                        val = val["__value__"]
                    call_kwargs[name] = val

        result = func(**call_kwargs)

        return _normalize_result(result, spec)


def _normalize_result(result: Any, spec: OmniSpec) -> Tuple[Any, ...]:

    n = len(spec.return_types)
    if n == 0:
        return ()
    if isinstance(result, tuple):
        vals = list(result)
    elif isinstance(result, list):
        vals = list(result)
    elif isinstance(result, dict) and "result" in result:
        r = result["result"]
        vals = list(r) if isinstance(r, (tuple, list)) else [r]
    else:
        vals = [result]

    if len(vals) < n:
        vals = vals + [None] * (n - len(vals))
    elif len(vals) > n:
        vals = vals[:n]
    return tuple(vals)


# Omni Node Guide

import json as _json
import os as _os
import random as _random
import threading as _threading

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None


_OMNI_GUIDE_SYSTEM_PROMPT_CACHE: Dict[str, str] = {"text": "", "mtime": 0.0}


def _omni_guide_system_prompt_path() -> str:
    here = _os.path.dirname(_os.path.abspath(__file__))
    return _os.path.join(here, "workflows", "omni_node_system_prompt.txt")


def _load_omni_guide_system_prompt() -> str:
    path = _omni_guide_system_prompt_path()
    try:
        st = _os.stat(path)
    except OSError:
        return _OMNI_GUIDE_SYSTEM_PROMPT_CACHE["text"] or ""
    mtime = st.st_mtime
    if _OMNI_GUIDE_SYSTEM_PROMPT_CACHE["mtime"] == mtime and _OMNI_GUIDE_SYSTEM_PROMPT_CACHE["text"]:
        return _OMNI_GUIDE_SYSTEM_PROMPT_CACHE["text"]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return _OMNI_GUIDE_SYSTEM_PROMPT_CACHE["text"] or ""
    _OMNI_GUIDE_SYSTEM_PROMPT_CACHE["text"] = text
    _OMNI_GUIDE_SYSTEM_PROMPT_CACHE["mtime"] = mtime
    return text


def _call_openrouter(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    service_tier: str,
    seed: int,
    temperature: float = 1.0,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Call OpenRouter chat completions and return the parsed JSON response.

    Mirrors the request shape used by ComfyUI-OpenRouter-Node, including the
    HTTP-Referer / X-Title headers and Connection: close session tweak that
    avoid ConnectionResetError 10054 from OpenRouter.
    """
    if _requests is None:
        raise RuntimeError("OmniNode: 'requests' package is required for OpenRouter calls.")
    if not api_key or not api_key.startswith("sk-or-"):
        raise RuntimeError(
            "OmniNode: a valid OpenRouter API key is required. Add an "
            "OmniNodeGuide node to the graph and set its api_key."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nova452/OmniNode",
        "X-Title": "ComfyUI Rebalance-Pack Omni Node",
        "User-Agent": "ComfyUI-Rebalance-Pack-OmniNode/1.0",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed & 0x7FFFFFFF,
    }
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    if service_tier and service_tier != "default":
        payload["service_tier"] = service_tier

    with _requests.Session() as session:
        session.headers.update({"Connection": "close"})
        response = session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


def _extract_content(result_json: Dict[str, Any]) -> str:
    try:
        return result_json["choices"][0]["message"].get("content", "") or ""
    except (KeyError, IndexError, TypeError):
        return ""


_OMNI_GENERATED_CODE: Dict[str, str] = {}
_OMNI_CODE_LOCK = _threading.Lock()


def _set_generated_code(node_id: str, code: str) -> None:
    with _OMNI_CODE_LOCK:
        _OMNI_GENERATED_CODE[str(node_id)] = code


def _get_generated_code(node_id: str) -> str:
    with _OMNI_CODE_LOCK:
        return _OMNI_GENERATED_CODE.get(str(node_id), "")


# Server routes

def _register_omni_build_route() -> None:
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return

    server = PromptServer.instance
    if getattr(server, "_rebalance_omni_build_route_registered", False):
        return
    server._rebalance_omni_build_route_registered = True

    routes = web.RouteTableDef()

    @routes.post("/rebalance_pack/omni_build")
    async def _omni_build(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body."}, status=400)

        node_id = str(body.get("node_id", ""))
        user_prompt = str(body.get("user_prompt", ""))

        if not node_id:
            return web.json_response({"error": "Missing node_id."}, status=400)
        if not user_prompt.strip():
            return web.json_response({"error": "Missing user_prompt."}, status=400)


        api_key = ""
        model = "openai/gpt-5.6-luna"
        reasoning_effort = "none"
        service_tier = "default"
        system_prompt = ""

        prompt_json = body.get("prompt")
        # graphToPrompt() returns {output: {...}, workflow: {...}}; the node
        # data lives under the "output" key. Fall back to the top-level dict
        # for older formats or direct node dicts.
        if isinstance(prompt_json, dict) and isinstance(prompt_json.get("output"), dict):
            prompt_json = prompt_json["output"]
        if isinstance(prompt_json, dict):
            for nid, ndata in prompt_json.items():
                if not isinstance(ndata, dict):
                    continue
                if ndata.get("class_type") != "OmniNodeGuide":
                    continue
                ginputs = ndata.get("inputs", {})
                if not isinstance(ginputs, dict):
                    continue
                api_key = str(ginputs.get("api_key", "") or "")
                model = str(ginputs.get("model", model) or model)
                reasoning_effort = str(ginputs.get("reasoning_effort", reasoning_effort) or reasoning_effort)
                service_tier = str(ginputs.get("service_tier", service_tier) or service_tier)
                sp = ginputs.get("system_prompt")
                if isinstance(sp, str):
                    system_prompt = sp
                break

        if not system_prompt.strip():
            system_prompt = _load_omni_guide_system_prompt()

        seed = body.get("seed", None)
        if seed is None:
            seed = _random.randint(0, 0x7FFFFFFF)
        else:
            try:
                seed = int(seed) & 0x7FFFFFFF
            except (TypeError, ValueError):
                seed = _random.randint(0, 0x7FFFFFFF)

        try:
            result_json = _call_openrouter(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                seed=seed,
            )
        except Exception as e:
            logger.exception("OmniNodeAPI: OpenRouter call failed")
            return web.json_response({"error": f"OpenRouter request failed: {e}"}, status=502)

        code = _extract_content(result_json)
        if not code:
            return web.json_response({"error": "Empty response from model."}, status=502)

        code = _strip_code_fences(code)
        _set_generated_code(node_id, code)

        try:
            PromptServer.instance.send_sync("rebalance_omni_built", {
                "node_id": node_id,
                "code": code,
                "seed": seed,
            })
        except Exception:
            pass

        return web.json_response({"code": code, "seed": seed})

    if not server.app.router.frozen:
        try:
            server.app.add_routes(routes)
        except Exception:
            logger.exception("Rebalance-Pack: failed to register /rebalance_pack/omni_build route")


def _strip_code_fences(text: str) -> str:

    import re as _re
    s = text.strip()

    m = _re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```", s, _re.DOTALL)
    if m:
        return m.group(1).strip()

    if not s.startswith("```"):
        return s
    nl = s.find("\n")
    if nl == -1:
        return s
    s = s[nl + 1:]
    if s.rstrip().endswith("```"):
        s = s.rstrip()
        s = s[:-3]
    return s.strip()


_register_omni_build_route()


class OmniNodeGuide:

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "sk-or-v1-...",
                    "multiline": False,
                    "tooltip": "Supported: OpenRouter",
                }),
                "model": ("STRING", {
                    "default": "openai/gpt-5.6-luna",
                }),
                "reasoning_effort": (["none", "minimal", "low", "medium", "high", "xhigh"], {
                    "default": "none",
                }),
                "service_tier": (["default", "flex", "priority"], {
                    "default": "default",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES: Tuple[Any, ...] = ()
    RETURN_NAMES: Tuple[str, ...] = ()
    FUNCTION = "execute"
    CATEGORY = "Rebalance-Pack/foundational"
    OUTPUT_IS_LIST: Tuple[bool, ...] = ()

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> bool:
        return True

    def execute(self, api_key: str, model: str,
                reasoning_effort: str, service_tier: str,
                system_prompt: str = "", unique_id=None, prompt=None, **kwargs) -> Tuple[Any, ...]:
        if isinstance(unique_id, list) and unique_id:
            unique_id = unique_id[0]
        node_id = str(unique_id) if unique_id is not None else ""

        if not system_prompt or not system_prompt.strip():
            system_prompt = _load_omni_guide_system_prompt()

        user_prompt = ""
        or_node_id = ""
        if isinstance(prompt, dict):
            for nid, ndata in prompt.items():
                if not isinstance(ndata, dict):
                    continue
                if ndata.get("class_type") != "OmniNodeAPI":
                    continue
                ninputs = ndata.get("inputs", {})
                if not isinstance(ninputs, dict):
                    continue
                up = ninputs.get("prompt")
                if isinstance(up, str) and up.strip():
                    user_prompt = up
                    or_node_id = nid
                    break

        if not user_prompt.strip():
            raise RuntimeError(
                "OmniNodeGuide: no prompt found. Add an OmniNodeAPI node "
                "to the graph and enter a prompt."
            )

        seed = _random.randint(0, 0x7FFFFFFF)

        result_json = _call_openrouter(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            seed=seed,
        )
        code = _strip_code_fences(_extract_content(result_json))
        if not code:
            raise RuntimeError("OmniNodeGuide: model returned empty content.")


        _set_generated_code(or_node_id or node_id, code)

        try:
            from server import PromptServer
            PromptServer.instance.send_sync("rebalance_omni_built", {
                "node_id": or_node_id or node_id,
                "code": code,
                "seed": seed,
            })
        except Exception:
            pass

        return ()


class OmniNodeAPI:
    """Omni Node (OpenRouter)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_prompt": "PROMPT",
                "omni_code": "STRING",
            },
        }

    _WILDCARD_OUT = tuple("*" for _ in range(24))
    RETURN_TYPES = _WILDCARD_OUT
    RETURN_NAMES = tuple(f"output_{i + 1}" for i in range(24))
    FUNCTION = "execute"
    CATEGORY = "Rebalance-Pack/foundational"
    OUTPUT_IS_LIST = tuple(False for _ in range(24))

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> bool:
        return True

    def execute(self, prompt: str = "",
                unique_id=None, extra_prompt=None,
                omni_code=None, **kwargs) -> Tuple[Any, ...]:
        if isinstance(unique_id, list) and unique_id:
            unique_id = unique_id[0]
        node_id = str(unique_id) if unique_id is not None else ""


        code = _get_generated_code(node_id)
        if not code and omni_code:
            if isinstance(omni_code, list) and omni_code:
                omni_code = omni_code[0]
            code = omni_code or ""

        cls_gen, spec = compile_omni_class(code or "")
        if cls_gen is None or spec.error:
            raise RuntimeError(f"OmniNodeAPI: {spec.error or 'no generated code yet — press Build.'}")

        func_name = spec.function or "execute"
        func = getattr(cls_gen(), func_name, None)
        if func is None:
            raise RuntimeError(
                f"OmniNodeAPI: generated class {spec.class_name!r} has no function "
                f"{func_name!r}."
            )

        declared = set(spec.required.keys()) | set(spec.optional.keys())
        call_kwargs: Dict[str, Any] = {}

        for k, v in kwargs.items():
            if k in declared:
                call_kwargs[k] = v

        if extra_prompt and node_id:
            node_inputs = None
            try:
                node_inputs = extra_prompt[node_id]["inputs"]
            except (KeyError, TypeError):
                node_inputs = None
            if isinstance(node_inputs, dict):
                for name in declared:
                    if name in call_kwargs:
                        continue
                    if name not in node_inputs:
                        continue
                    val = node_inputs[name]
                    if isinstance(val, list) and len(val) == 2 and all(isinstance(x, (int, str)) for x in val):
                        continue
                    if isinstance(val, dict) and "__value__" in val:
                        val = val["__value__"]
                    call_kwargs[name] = val

        result = func(**call_kwargs)
        return _normalize_result(result, spec)
