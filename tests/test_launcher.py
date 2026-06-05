from pathlib import Path


def test_windows_launcher_starts_gui_from_project_root():
    launcher = Path("start_tmm_device_sim.bat")

    assert launcher.exists()
    text = launcher.read_text(encoding="utf-8")
    assert 'cd /d "%~dp0"' in text
    assert "PYTHONPATH" in text
    assert "\\src" in text
    assert "-m tmm_device_sim" in text
