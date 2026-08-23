"""Ranking files against a query, and the caches that make it quick.

Two indexes, because there are two ways to search: the file's own content, and
the text of its description. Both compare in the same vector space, so a typed
query can rank a photo, a track and a document against each other.
"""
