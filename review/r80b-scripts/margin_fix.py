import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth
synth.PR = synth.load(os.environ.get("FIX", "fixcap"))
import margin_real
margin_real.PR = synth.PR
margin_real.main()
