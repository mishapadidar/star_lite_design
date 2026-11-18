#!/usr/bin/env python3
"""Run a python script with ANOTHER build of the compiled simsoptpp module.

    avx512_loader.py <simsoptpp .so> <script> [args of the script...]

The simsoptpp of a venv is found by the finder of its editable install, which comes before
PYTHONPATH, so another build cannot be selected with PYTHONPATH. Here the given .so is loaded
first and registered as the module `simsoptpp`; every later `import simsoptpp` (simsopt does
it) then gets that one. The script is run unchanged, as __main__, with its own arguments.
Used by run_boozer_all.sh and run_double.sh for the AVX-512 build on the icelake nodes."""
import importlib.machinery
import importlib.util
import runpy
import sys

so, script = sys.argv[1], sys.argv[2]
loader = importlib.machinery.ExtensionFileLoader('simsoptpp', so)
spec = importlib.util.spec_from_file_location('simsoptpp', so, loader=loader)
module = importlib.util.module_from_spec(spec)
sys.modules['simsoptpp'] = module
spec.loader.exec_module(module)
if script == '--check':          # only say which file was loaded (used as a load test)
    import simsoptpp
    print(simsoptpp.__file__)
    sys.exit(0 if simsoptpp.__file__ == so else 1)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name='__main__')
