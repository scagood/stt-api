import asyncio, sys, os
sys.path.insert(0, os.getcwd())
import tests.conftest
from types import SimpleNamespace
from starlette.datastructures import FormData
from fastapi import HTTPException
from parakeet_service import routes

async def go(pairs):
    async def form():
        return FormData(pairs)
    try:
        await routes._no_aligner_quantization(SimpleNamespace(form=form))
        return None
    except HTTPException as e:
        return e.status_code, e.detail

for pairs in [
    [("aligner", "wav2vec2-base-960h"), ("aligner_quantization", "fp32")],
    [("aligner", "mms-300m-forced-aligner"), ("aligner_quantization", "int8")],
    [("aligner", "wav2vec2-base-960h:int8"), ("aligner_quantization", " FP16 ")],
    [("aligner_quantization", "fp32")],
    [("aligner", ""), ("aligner_quantization", "fp32")],
    [("aligner", "wav2vec2-base-960h"), ("aligner_quantization", "")],
]:
    print(pairs, "->", asyncio.run(go(pairs)))
for v in ["wav2vec2-base-960h:", "wav2vec2-base-960h:FP16"]:
    try:
        print(v, routes._named(v, "aligner"))
    except HTTPException as e:
        print(v, "->", e.status_code, e.detail)
