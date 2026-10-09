import sys, importlib.util
S="/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path + "/parakeet_service/__init__.py", submodule_search_locations=[path + "/parakeet_service"])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    spec2 = importlib.util.spec_from_file_location(name + ".chunker", path + "/parakeet_service/chunker.py")
    m = importlib.util.module_from_spec(spec2); sys.modules[name + ".chunker"] = m; spec2.loader.exec_module(m)
    return m
pr = load("pr_pkg", S + "/wf71")
fx = load("fx_pkg", S + "/wf71-scripts/fixF1")
spec = importlib.util.spec_from_file_location("pr_pkg.chunker_main", S + "/wf71-scripts/main_chunker.py")
mn = importlib.util.module_from_spec(spec); spec.loader.exec_module(mn)
SR = 16000
