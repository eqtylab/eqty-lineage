import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("t", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
PROBES = ["  Hello, World!  ", "a_b  c", "Rock & Roll?", "-hello-", "Café Ω x",
          "abc 123", "a---b", "", "!!!", "well-known thing", "UPPER   lower", "tabs\there"]
out = []
for p in PROBES:
    try: out.append(repr(m.slugify(p)))
    except Exception as e: out.append(f"<{type(e).__name__}>")
print(json.dumps(out))
