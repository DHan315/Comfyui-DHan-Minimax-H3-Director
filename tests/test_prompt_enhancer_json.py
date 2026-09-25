import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_enhancer():
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(
            routes=types.SimpleNamespace(get=lambda *_a, **_k: lambda fn: fn)
        )
    )
    server.web = types.SimpleNamespace(json_response=lambda *a, **k: (a, k))
    sys.modules.setdefault("server", server)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: ""
    sys.modules.setdefault("folder_paths", folder_paths)

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = types.SimpleNamespace(
        ComfyNode=object,
        String=types.SimpleNamespace(Input=lambda *a, **k: None, Output=lambda *a, **k: None),
        Combo=types.SimpleNamespace(Input=lambda *a, **k: None),
        Int=types.SimpleNamespace(Input=lambda *a, **k: None),
        Float=types.SimpleNamespace(Input=lambda *a, **k: None),
        Boolean=types.SimpleNamespace(Input=lambda *a, **k: None),
        Image=types.SimpleNamespace(Input=lambda *a, **k: None),
    )
    sys.modules.setdefault("comfy_api", comfy_api)
    sys.modules.setdefault("comfy_api.latest", latest)

    path = Path(__file__).resolve().parents[1] / "dhan_h3_prompt_enhancer.py"
    spec = importlib.util.spec_from_file_location("dhan_h3_prompt_enhancer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OllamaResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enhancer = load_enhancer()

    def test_empty_response_has_ollama_metadata(self):
        with self.assertRaisesRegex(ValueError, "empty response content.*done_reason"):
            self.enhancer._extract_json(
                "", source="Ollama response", response_summary='{"done_reason":"load"}'
            )

    def test_invalid_response_has_preview(self):
        with self.assertRaisesRegex(ValueError, "not valid JSON.*hello"):
            self.enhancer._extract_json("hello from model")

    def test_fenced_json_still_parses(self):
        parsed = self.enhancer._extract_json('```json\n{"global_prompt":"x"}\n```')
        self.assertEqual(parsed["global_prompt"], "x")

    def test_summary_excludes_generated_content(self):
        summary = self.enhancer._ollama_response_summary({
            "done_reason": "stop", "message": {"content": "generated storyboard"}
        })
        self.assertIn("done_reason", summary)
        self.assertNotIn("generated storyboard", summary)

    def test_empty_chat_retries_without_json_then_generate(self):
        enhancer = self.enhancer
        sent = []
        enhancer.server.PromptServer.instance.send_sync = lambda event, result: sent.append(result)
        enhancer.DHanH3PromptEnhancer.hidden = types.SimpleNamespace(unique_id="test-node")
        enhancer.io.NodeOutput = lambda: None

        class Response:
            status = 200

            def __init__(self, data):
                self.data = data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                pass

            async def text(self):
                return json.dumps(self.data)

        class Session:
            def __init__(self):
                self.posts = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                pass

            def get(self, *_args, **_kwargs):
                return Response({"models": []})

            def post(self, endpoint, json=None):
                self.posts.append((endpoint, json))
                if len(self.posts) < 3:
                    return Response({"message": {"content": ""}})
                return Response({"response": '{"global_prompt":"scene","storyboard":[{"prompt":"move"}]}'})

        session = Session()

        async def skip_unload(*_args):
            pass

        with patch.object(enhancer.aiohttp, "ClientSession", return_value=session), \
                patch.object(enhancer, "_force_unload", skip_unload):
            asyncio.run(enhancer.DHanH3PromptEnhancer.execute(
                idea="A person moves", parts=1, duration_seconds=5,
                ollama_model="test-model", timing_mode="Equal"
            ))

        self.assertEqual([url.rsplit("/", 1)[-1] for url, _ in session.posts],
                         ["chat", "chat", "generate"])
        self.assertEqual(session.posts[0][1]["think"], False)
        self.assertEqual(session.posts[0][1]["options"]["num_predict"], 8192)
        self.assertNotIn("format", session.posts[1][1])
        self.assertEqual(session.posts[2][1]["think"], False)
        self.assertEqual(sent[0]["global_prompt"], "scene")


if __name__ == "__main__":
    unittest.main()
