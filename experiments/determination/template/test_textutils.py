from textutils import slugify, titlecase


def test_titlecase():
    assert titlecase("hello world") == "Hello World"


def test_slugify_basic():
    assert slugify("  Hello, World!  ") == "hello-world"


def test_slugify_underscores():
    assert slugify("a_b  c") == "a-b-c"


def test_slugify_punctuation():
    assert slugify("Rock & Roll?") == "rock-roll"
