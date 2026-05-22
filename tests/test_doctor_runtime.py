from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error as urllib_error

from src.agent.agent_types import AgentRuntimeConfig, ModelConfig
from src.features.system.doctor_runtime import run_doctor
from tests.test_helpers import FakeHTTPResponse


class DoctorRuntimeTests(unittest.TestCase):
    def test_run_doctor_reports_backend_and_model_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.git').mkdir()
            runtime_config = AgentRuntimeConfig(
                cwd=workspace,
                session_directory=workspace / '.port_sessions' / 'agent',
                scratchpad_root=workspace / '.port_sessions' / 'scratchpad',
            )
            model_config = ModelConfig(
                model='demo-model',
                base_url='http://127.0.0.1:8000/v1',
                api_key='local-token',
                timeout_seconds=1.0,
            )
            with patch(
                'src.features.system.doctor_runtime.urllib_request.urlopen',
                return_value=FakeHTTPResponse({'data': [{'id': 'demo-model'}]}),
            ), patch(
                'src.features.system.doctor_runtime.importlib.util.find_spec',
                return_value=object(),
            ), patch(
                'src.features.system.doctor_runtime.shutil.which',
                return_value='C:/Program Files/Git/bin/git.exe',
            ):
                report = run_doctor(
                    model_config=model_config,
                    runtime_config=runtime_config,
                )

        self.assertFalse(report.has_failures)
        rendered = report.as_text()
        self.assertIn('[ok] workspace:', rendered)
        self.assertIn('[ok] session-storage:', rendered)
        self.assertIn('[ok] backend:', rendered)
        self.assertIn('[ok] model:', rendered)
        self.assertIn('ready=yes', rendered)

    def test_run_doctor_marks_backend_unreachable_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            runtime_config = AgentRuntimeConfig(cwd=workspace)
            model_config = ModelConfig(
                model='demo-model',
                base_url='http://127.0.0.1:8000/v1',
                api_key='local-token',
                timeout_seconds=1.0,
            )
            with patch(
                'src.features.system.doctor_runtime.urllib_request.urlopen',
                side_effect=urllib_error.URLError('connection refused'),
            ):
                report = run_doctor(
                    model_config=model_config,
                    runtime_config=runtime_config,
                    check_tui=False,
                )

        self.assertTrue(report.has_failures)
        self.assertIn('[fail] backend:', report.as_text())

    def test_run_doctor_warns_when_pillow_missing_for_tui_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            runtime_config = AgentRuntimeConfig(cwd=workspace)
            model_config = ModelConfig(model='demo-model')

            def fake_find_spec(name: str) -> object | None:
                if name == 'textual.app':
                    return object()
                if name == 'PIL.Image':
                    return None
                return None

            with patch(
                'src.features.system.doctor_runtime.importlib.util.find_spec',
                side_effect=fake_find_spec,
            ):
                report = run_doctor(
                    model_config=model_config,
                    runtime_config=runtime_config,
                    check_backend=False,
                )

        rendered = report.as_text()
        self.assertFalse(report.has_failures)
        self.assertIn('[warn] tui:', rendered)
        self.assertIn('Pillow is missing', rendered)


if __name__ == '__main__':
    unittest.main()
