import asyncio, sys
sys.path.insert(0, "."); sys.path.insert(0, "tests")
import conftest
from starlette.requests import Request
from fastapi import HTTPException
from parakeet_service import routes

def req(body, ctype):
    sent = False
    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}
    scope = {"type": "http", "method": "POST", "path": "/", "headers": [(b"content-type", ctype.encode()), (b"content-length", str(len(body)).encode())], "query_string": b""}
    return Request(scope, receive)

async def main():
    b = "XyZ"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"aligner\"\r\n\r\nwav2vec2-base-960h\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"aligner_quantization\"\r\n\r\n\r\n--{b}--\r\n").encode()
    for r in [req(body, f"multipart/form-data; boundary={b}"), req(b"aligner=wav2vec2-base-960h&aligner_quantization=", "application/x-www-form-urlencoded")]:
        try:
            await routes._no_aligner_quantization(r)
            print("no error")
        except HTTPException as e:
            print(e.status_code, e.detail)
asyncio.run(main())
