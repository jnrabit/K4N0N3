from k4n0n3.cache import LRUCache

def test_put_get():
    cache = LRUCache[int](max_size=3)
    cache.put("a", 1)
    assert cache.get("a") == 1

def test_eviction():
    cache = LRUCache[int](max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_lru_order():
    cache = LRUCache[int](max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1

def test_ttl_expired():
    cache = LRUCache[int](max_size=5, ttl=0.0)
    cache.put("a", 1)
    assert cache.get("a") is None

def test_contains():
    cache = LRUCache[int](max_size=5)
    cache.put("x", 42)
    assert "x" in cache
    assert "y" not in cache

def test_remove():
    cache = LRUCache[int](max_size=5)
    cache.put("a", 1)
    cache.remove("a")
    assert cache.get("a") is None

def test_clear():
    cache = LRUCache[int](max_size=5)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.clear()
    assert len(cache) == 0

def test_contains_ttl_expired():
    cache = LRUCache[int](max_size=5, ttl=0.0)
    cache.put("a", 1)
    assert "a" not in cache

def test_len():
    cache = LRUCache[int](max_size=5)
    assert len(cache) == 0
    cache.put("a", 1)
    assert len(cache) == 1
