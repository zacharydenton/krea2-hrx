import ctypes as C, unicodedata as u, hashlib, random
from tokenizers import Tokenizer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
lib = C.CDLL(str(ROOT / "build/libtokenizer_test.so"))
lib.tok_create.argtypes = [C.c_char_p]
lib.tok_create.restype = C.c_void_p
lib.tok_free.argtypes = [C.c_void_p]
lib.tok_encode.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int32)]
lib.nfc.argtypes = [C.c_char_p, C.c_char_p]
lib.hash.argtypes = [C.c_char_p, C.c_size_t, C.c_char_p]
path = str(ROOT / "build/native/tokenizer.json")
tok = lib.tok_create(path.encode())
assert tok
ref = Tokenizer.from_file(path)
out = (C.c_int32 * 65536)()
buf = C.create_string_buffer(100000)
for n in [0, 1, 55, 56, 63, 64, 65, 1000, 1000000]:
    data = bytes((i * 37) % 256 for i in range(n))
    lib.hash(data, n, buf)
    assert buf.value.decode() == hashlib.sha256(data).hexdigest(), n
count = 0
for cp in range(0x110000):
    if u.decomposition(chr(cp)):
        s = chr(cp)
        n = lib.nfc(s.encode(), buf)
        assert n >= 0
        assert buf.value.decode() == u.normalize("NFC", s), (hex(cp), buf.value)
        count += 1
samples = [
    "a red fox in the snow",
    "cafe\u0301",
    "a\u0301b",
    "தமிழ் ௧௨௩",
    "hello <|im_start|>world",
    "\u1100\u1161\u11a8",
    " I'ſ here",
    "   leading   and trailing  ",
    "\n \r\n",
    "q\u0307\u0323",
]
rng = random.Random(4)
alphabet = "aZ é\u0301\u0323\u1100\u1161\u11a8\t\n123.,'\"中🦊௧௨௩αß"
samples += ["".join(rng.choices(alphabet, k=rng.randrange(1, 80))) for _ in range(300)]
for s in samples:
    n = lib.tok_encode(tok, s.encode(), out)
    want = ref.encode(s, add_special_tokens=False).ids
    assert n >= 0 and list(out[:n]) == want, (repr(s), list(out[:n]), want)

for invalid in [
    b"\xff",
    b"\xc0\x80",
    b"\xed\xa0\x80",
    b"\xf4\x90\x80\x80",
    b"\xe2\x82",
    b"\xe2xx",
]:
    assert lib.tok_encode(tok, invalid, out) == -1
lib.tok_free(tok)
print(
    "PASS SHA-256 vectors; NFC",
    count,
    "decomposable codepoints; tokenizer",
    len(samples),
    "mixed-script/special/combining prompts",
)
