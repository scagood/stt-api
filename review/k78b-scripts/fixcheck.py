"""Run repro_retime (TAIL 1.6) and edges-style random check against the fix worktree."""
import sys, runpy
import lib
lib.ROOTS["fix"] = f"{lib.S}/k78b-fix"
CF, RF = lib.load("fix")
lib.MODS["pr"] = (CF, RF)  # swap the PR for the fix
sys.argv = ["repro_retime.py", "1.6", "-42", "-48"]
runpy.run_path("repro_retime.py", run_name="__main__")
