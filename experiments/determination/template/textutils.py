"""Small text helpers."""

import re


def titlecase(value: str) -> str:
    """Capitalise each word, leaving existing capitals alone."""
    return " ".join(w[:1].upper() + w[1:] for w in value.split(" "))


def slugify(value: str) -> str:
    """Convert a string to a lowercase, hyphen-separated slug.

    Strips leading and trailing whitespace, lowercases, replaces any run of
    whitespace or underscores with a single hyphen, and removes characters
    that are not alphanumeric or hyphens.

        >>> slugify("  Hello, World!  ")
        'hello-world'
    """
    raise NotImplementedError
