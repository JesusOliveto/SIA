import importlib


def test_app_import_smoke():
    # Verifica que la app cargue sin errores (depende de winequality.arff presente)
    import app  # noqa: F401

    importlib.reload(app)
